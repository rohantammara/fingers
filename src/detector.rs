pub mod hand_detector {
    use anyhow::Result;
    use image::{ImageBuffer, Rgb, imageops::FilterType};
    use ndarray::{Array4, ArrayView, Ix3};
    use ort::{inputs, session::Session, session::builder::GraphOptimizationLevel, value::Value};
    use std::path::Path;

    const INPUT_SIZE: f32 = 256.0;
    const NUM_ANCHORS: usize = 2944;

    pub struct HandDetector {
        session: Session,
        anchors: Vec<Anchor>,
    }

    struct Anchor {
        x_center: f32,
        y_center: f32,
        w: f32,
        h: f32,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Box {
        pub xmin: f32,
        pub ymin: f32,
        pub xmax: f32,
        pub ymax: f32,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Landmark {
        pub x: f32,
        pub y: f32,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct HandDetails {
        pub score: f32,
        pub bbox: Box,
        pub wrist: Landmark,
    }

    fn generate_anchors(num_anchors: usize) -> Vec<Anchor> {
        let mut anchors = Vec::with_capacity(num_anchors);

        let strides = [8, 16, 32, 32];
        let map_sizes = [32, 16, 8, 8];
        let anchors_per_cell = [2, 2, 2, 6];

        for i in 0..strides.len() {
            let stride = strides[i] as f32;
            let map_size = map_sizes[i];

            for y in 0..map_size {
                for x in 0..map_size {
                    for _ in 0..anchors_per_cell[i] {
                        anchors.push(Anchor {
                            // Normalize to 0 to 1
                            x_center: (x as f32 + 0.5) * stride / INPUT_SIZE,
                            y_center: (y as f32 + 0.5) * stride / INPUT_SIZE,
                            w: 1.0,
                            h: 1.0,
                        });
                    }
                }
            }
        }
        anchors
    }

    fn get_bbox(
        best_score_idx: usize,
        coords: &ndarray::ArrayView3<f32>,
        anchors: &[Anchor],
    ) -> Box {
        // Extract the raw regression values
        let dx = coords[[0, best_score_idx, 0]];
        let dy = coords[[0, best_score_idx, 1]];
        let dw = coords[[0, best_score_idx, 2]];
        let dh = coords[[0, best_score_idx, 3]];

        // Get the corresponding anchor
        let anchor = &anchors[best_score_idx];

        // Apply the MediaPipe Scale (Standard is 256.0 for 256x256 input)
        // This transforms the raw offsets into normalized coordinates (0.0 to 1.0)
        let center_x = (dx / INPUT_SIZE) * anchor.w + anchor.x_center;
        let center_y = (dy / INPUT_SIZE) * anchor.h + anchor.y_center;
        let w = (dw / INPUT_SIZE) * anchor.w;
        let h = (dh / INPUT_SIZE) * anchor.h;

        // Return as a Bounding Box (top-left and bottom-right)
        Box {
            xmin: center_x - (w / 2.0),
            ymin: center_y - (h / 2.0),
            xmax: center_x + (w / 2.0),
            ymax: center_y + (h / 2.0),
        }
    }

    fn get_landmark(
        best_score_idx: usize,
        coords: &ndarray::ArrayView3<f32>,
        anchors: &[Anchor],
        coords_x_idx: usize,
        coords_y_idx: usize,
    ) -> Landmark {
        // Extract the raw regression values
        let x = coords[[0, best_score_idx, coords_x_idx]];
        let y = coords[[0, best_score_idx, coords_y_idx]];

        // Get the corresponding anchor
        let anchor = &anchors[best_score_idx];

        // Apply the MediaPipe Scale (Standard is 256.0 for 256x256 input)
        // This transforms the raw offsets into normalized coordinates (0.0 to 1.0)
        let landmark_x = (x / INPUT_SIZE) * anchor.w + anchor.x_center;
        let landmark_y = (y / INPUT_SIZE) * anchor.h + anchor.y_center;

        Landmark {
            x: landmark_x,
            y: landmark_y,
        }
    }

    fn intersection_over_union(box_a: &Box, box_b: &Box) -> f32 {
        let xmin = box_a.xmin.max(box_b.xmin);
        let ymin = box_a.ymin.max(box_b.ymin);
        let xmax = box_a.xmax.min(box_b.xmax);
        let ymax = box_a.ymax.min(box_b.ymax);

        let intersection_area = (xmax - xmin).max(0.0) * (ymax - ymin).max(0.0);

        let area_a = (box_a.xmax - box_a.xmin) * (box_a.ymax - box_a.ymin);
        let area_b = (box_b.xmax - box_b.xmin) * (box_b.ymax - box_b.ymin);

        intersection_area / (area_a + area_b - intersection_area)
    }

    fn apply_nms(mut candidates: Vec<HandDetails>, iou_threshold: f32) -> Vec<HandDetails> {
        // Sort scores by descending order
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        let mut selected = Vec::new();
        while !candidates.is_empty() {
            let best = candidates.remove(0);

            // Remove all other boxes that overlap too much with the 'best' box
            candidates
                .retain(|item| intersection_over_union(&best.bbox, &item.bbox) < iou_threshold);

            selected.push(best);
        }
        selected
    }

    impl HandDetector {
        pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
            // Create new session for model
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(4)?
                .commit_from_file(model_path)?;

            // Generate array of all anchors
            let anchors = generate_anchors(NUM_ANCHORS);

            Ok(Self { session, anchors })
        }

        pub fn new_embedded(model_bytes: &[u8]) -> Result<Self> {
            // Create new session for model
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(4)?
                .commit_from_memory(model_bytes)?;

            // Generate array of all anchors
            let anchors = generate_anchors(NUM_ANCHORS);

            Ok(Self { session, anchors })
        }

        pub fn detect(
            &mut self,
            frame: &ImageBuffer<Rgb<u8>, Vec<u8>>,
        ) -> Result<Option<Vec<HandDetails>>> {
            // Preprocessing: Resize the image to what the model expects
            // (letterbox image to fix aspect ratio without cropping or stretching)

            let target_size = INPUT_SIZE as u32;

            let mut canvas = ImageBuffer::from_pixel(target_size, target_size, Rgb([0, 0, 0]));

            let (frame_width, frame_height) = frame.dimensions();
            let scale = target_size as f32 / frame_width as f32;
            let new_height = (frame_height as f32 * scale) as u32;

            let resized_frame =
                image::imageops::resize(frame, target_size, new_height, FilterType::Triangle);

            let top_padding = (target_size - new_height) / 2;
            image::imageops::overlay(&mut canvas, &resized_frame, 0, top_padding as i64);

            // Create closure to normalize the y coordinates to the original frame aspect ratio (since only black bars added to y-dimension)
            let normalized_padding = top_padding as f32 / target_size as f32;
            let normalized_content_height = new_height as f32 / target_size as f32;

            let norm_y = |y: f32| -> f32 { (y - normalized_padding) / normalized_content_height };

            // Convert Image to Tensor [1, 3, 256, 256]
            // We normalize pixels from 0-255 (u8) to 0.0-1.0 (f32)
            let mut input =
                Array4::<f32>::zeros((1, 3, target_size as usize, target_size as usize));

            for (x, y, rgb) in canvas.enumerate_pixels() {
                input[[0, 0, y as usize, x as usize]] = rgb[0] as f32 / 255.0; // R
                input[[0, 1, y as usize, x as usize]] = rgb[1] as f32 / 255.0; // G
                input[[0, 2, y as usize, x as usize]] = rgb[2] as f32 / 255.0; // B
            }

            let input_tensor = Value::from_array(input)?;

            // Run Inference
            let outputs = self.session.run(inputs!["image" => input_tensor])?;

            // Output processing
            // Extract both output tensors
            let (scores_shape, scores_data) = outputs["box_scores"].try_extract_tensor::<f32>()?;
            let (coords_shape, coords_data) = outputs["box_coords"].try_extract_tensor::<f32>()?;

            let scores_shape_usize: Vec<usize> = scores_shape.iter().map(|&x| x as usize).collect();
            let coords_shape_usize: Vec<usize> = coords_shape.iter().map(|&x| x as usize).collect();

            let scores = ArrayView::from_shape(scores_shape_usize, scores_data)?
                .into_dimensionality::<Ix3>()?;
            let coords = ArrayView::from_shape(coords_shape_usize, coords_data)?
                .into_dimensionality::<Ix3>()?;

            let num_anchors = scores.shape()[1];

            let mut candidates = Vec::new();
            let score_threshold = 1.0 as f32;
            let nms_iou_threshold = 0.3;

            for i in 0..num_anchors {
                let score = scores[[0, i, 0]];
                if score > score_threshold {
                    let mut bbox = get_bbox(i, &coords, &self.anchors);
                    let mut wrist = get_landmark(i, &coords, &self.anchors, 4, 5);

                    [bbox.ymin, bbox.ymax, wrist.y] =
                        [bbox.ymin, bbox.ymax, wrist.y].map(|y| norm_y(y));

                    candidates.push(HandDetails {
                        score: score,
                        bbox: bbox,
                        wrist: wrist,
                    });
                }
            }

            // Get best candidates based on Non-Maximum Suppression
            let filtered_hands = apply_nms(candidates, nms_iou_threshold);

            if filtered_hands.is_empty() {
                println!("No hands detected");
                Ok(None)
            } else {
                // Return top 2 hands detected
                Ok(Some(
                    filtered_hands
                        .into_iter()
                        .take(2)
                        .collect::<Vec<HandDetails>>(),
                ))
            }
        }
    }
}
