pub mod hand_detector {
    use anyhow::Result;
    use image::{ImageBuffer, Rgb, imageops::FilterType};
    use ndarray::{Array4, ArrayView, Ix3};
    use ort::{inputs, session::Session, session::builder::GraphOptimizationLevel, value::Value};
    use std::path::Path;

    pub struct HandDetector {
        session: Session,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct BoundingBox {
        pub xmin: f32,
        pub ymin: f32,
        pub xmax: f32,
        pub ymax: f32,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct WristCoords {
        pub x: f32,
        pub y: f32,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct HandDetails {
        pub bbox: BoundingBox,
        pub wrist: WristCoords,
    }

    impl HandDetector {
        pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(4)?
                .commit_from_file(model_path)?;

            Ok(Self { session })
        }

        pub fn detect(
            &mut self,
            frame: &ImageBuffer<Rgb<u8>, Vec<u8>>,
        ) -> Result<Option<HandDetails>> {
            // Preprocessing: Resize the image to what the model expects
            // (letterbox image to fix aspect ratio without cropping or stretching)

            let target_size = 256;

            let mut canvas = ImageBuffer::from_pixel(target_size, target_size, Rgb([0, 0, 0]));

            let (frame_width, frame_height) = frame.dimensions();
            let scale = target_size as f32 / frame_width as f32;
            let new_height = (frame_height as f32 * scale) as u32;

            let resized_frame =
                image::imageops::resize(frame, target_size, new_height, FilterType::Triangle);

            let top_padding = (target_size - new_height) / 2;
            image::imageops::overlay(&mut canvas, &resized_frame, 0, top_padding as i64);

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

            // Find the anchor with the best score (Argmax over all candidates and classes)
            let mut max_score = 0.0;
            let mut best_idx = 0;

            let num_anchors = scores.shape()[1];

            for i in 0..num_anchors {
                let score = scores[[0, i, 0]];
                if score > max_score {
                    max_score = score;
                    best_idx = i;
                }
            }

            // Threshold check: If the score is too low, no hand is visible
            if max_score < 0.65 {
                println!("No hand detected");
                return Ok(None);
            }

            // Extract details for the best index //
            // Get bounding box and wrist coords
            let [mut ymin, xmin, mut ymax, xmax] = [0, 1, 2, 3].map(|i| coords[[0, best_idx, i]]);
            // Get the first keypoint (wrist)
            let [mut wrist_y, wrist_x] = [4, 5].map(|i| coords[[0, best_idx, i]]);

            println!(
                "Raw output: bbox: {} {} {} {} | wrist: {} {}",
                xmin, ymin, xmax, ymax, wrist_x, wrist_y
            );

            // Normalize the y coordinates to the original frame aspect ratio (since only black bars added to y-dimension)
            let normalized_padding = top_padding as f32 / target_size as f32;
            let normalized_content_height = new_height as f32 / target_size as f32;

            [ymin, ymax, wrist_y] =
                [ymin, ymax, wrist_y].map(|i| (i - normalized_padding) / normalized_content_height);

            // Return Hand Details
            Ok(Some(HandDetails {
                bbox: BoundingBox {
                    xmin: xmin,
                    ymin: ymin,
                    xmax: xmax,
                    ymax: ymax,
                },
                wrist: WristCoords {
                    x: wrist_x,
                    y: wrist_y,
                },
            }))
        }
    }
}

#[allow(unused)]
pub mod gesture_detector {
    use anyhow::Result;
    use image::{ImageBuffer, Rgb, imageops::FilterType};
    use ndarray::{Array4, ArrayView, Ix3};
    use ort::{inputs, session::Session, session::builder::GraphOptimizationLevel, value::Value};
    use std::{io::Error, path::Path};

    #[derive(Debug, PartialEq, Clone, Copy)]
    pub enum Gesture {
        Unknown,    // No clear gesture
        ClosedFist, // All fingers curled into palm
        OpenPalm,   // All fingers extended
        Pointer,    // Index finger extended
        ThumbsDown, // Thumb pointing down
        ThumbsUp,   // Thumb pointing up
        Victory,    // Index and Middle fingers extended (V-shape)
        Cool,       // Thumb, Index and Little fingers extended
    }

    #[derive(Debug, Clone, Copy)]
    pub struct GestureDetails {
        pub gesture: Gesture,
        pub x: f32,
        pub y: f32,
    }

    pub struct GestureDetector {
        session: Session,
    }

    impl GestureDetector {
        pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(4)?
                .commit_from_file(model_path)?;

            Ok(Self { session })
        }

        pub fn detect(&mut self, frame: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<(), Error> {
            // Preprocessing: Resize the image to what the model expects
            // (letterbox image to fix aspect ratio without cropping or stretching)

            let target_size = 224;

            let mut canvas = ImageBuffer::from_pixel(target_size, target_size, Rgb([0, 0, 0]));

            let (frame_width, frame_height) = frame.dimensions();
            let scale = target_size as f32 / frame_width as f32;
            let new_height = (frame_height as f32 * scale) as u32;

            let resized_frame =
                image::imageops::resize(frame, target_size, new_height, FilterType::Triangle);

            let top_padding = (target_size - new_height) / 2;
            image::imageops::overlay(&mut canvas, &resized_frame, 0, top_padding as i64);

            // Convert Image to Tensor [1, 3, 256, 256]
            // We normalize pixels from 0-255 (u8) to 0.0-1.0 (f32)
            let mut input =
                Array4::<f32>::zeros((1, 3, target_size as usize, target_size as usize));

            for (x, y, rgb) in canvas.enumerate_pixels() {
                input[[0, 0, y as usize, x as usize]] = rgb[0] as f32 / 255.0; // R
                input[[0, 1, y as usize, x as usize]] = rgb[1] as f32 / 255.0; // G
                input[[0, 2, y as usize, x as usize]] = rgb[2] as f32 / 255.0; // B
            }

            let input_tensor = Value::from_array(input);

            // Run Inference
            //let outputs = self.session.run(inputs!["image" => input_tensor]);

            // Output processing
            // TODO

            // Find the best detection (Argmax over all candidates and classes)

            // Threshold check: If the score is too low, no hand is visible

            // 5. Map class index to Gesture
            // let gesture = match best_class {
            //     0 => Gesture::Unknown,
            //     1 => Gesture::ClosedFist,
            //     2 => Gesture::Pointer,
            //     _ => Gesture::Unknown,
            // };

            // Normalize the y coordinates to the original frame aspect ratio (since only black bars added to y-dimension)
            let normalized_padding = top_padding as f32 / target_size as f32;
            let normalized_content_height = new_height as f32 / target_size as f32;

            // Return
            Ok(())
        }
    }
}
