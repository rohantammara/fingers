pub mod hand_tracker {
    use anyhow::Result;
    use image::{ImageBuffer, Rgb, imageops::FilterType};
    use ndarray::{Array4, ArrayView, Ix3};
    use ort::{inputs, session::Session, session::builder::GraphOptimizationLevel, value::Value};
    use std::path::Path;

    pub struct HandGestureDetector {
        session: Session,
    }

    #[derive(Debug, PartialEq, Clone, Copy)]
    pub enum Gesture {
        OpenPalm,
        ClosedFist,
        Pointer,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct DetailedGesture {
        pub gesture: Gesture,
        pub x: f32,
        pub y: f32,
        pub score: f32,
    }

    impl HandGestureDetector {
        pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(4)?
                .commit_from_file(model_path)?;

            // Print model input and output names for debugging
            println!("Model Inputs: {:?}", session.inputs());
            println!("Model Outputs: {:?}", session.outputs());

            Ok(Self { session })
        }

        pub fn detect(
            &mut self,
            frame: &ImageBuffer<Rgb<u8>, Vec<u8>>,
        ) -> Result<Option<DetailedGesture>> {
            // 1. Preprocessing: Resize the image to what the model expects (usually 224x224 or 256x256)
            // Check your model's documentation/metadata for the exact size.
            let target_size = 256;
            let resized =
                image::imageops::resize(frame, target_size, target_size, FilterType::Triangle);

            // 2. Convert Image to Tensor [1, 3, 256, 256]
            // We normalize pixels from 0-255 (u8) to 0.0-1.0 (f32)
            let mut input =
                Array4::<f32>::zeros((1, 3, target_size as usize, target_size as usize));

            for (x, y, rgb) in resized.enumerate_pixels() {
                input[[0, 0, y as usize, x as usize]] = rgb[0] as f32 / 255.0; // R
                input[[0, 1, y as usize, x as usize]] = rgb[1] as f32 / 255.0; // G
                input[[0, 2, y as usize, x as usize]] = rgb[2] as f32 / 255.0; // B
            }

            let input_tensor = Value::from_array(input)?;

            // 3. Run Inference
            let outputs = self.session.run(inputs!["image" => input_tensor])?;

            // 4. Output processing
            // Extract both output tensors
            // Use the names from your Netron image: "box_scores" and "box_coords"
            let (scores_shape, scores_data) = outputs["box_scores"].try_extract_tensor::<f32>()?;
            let (coords_shape, coords_data) = outputs["box_coords"].try_extract_tensor::<f32>()?;

            let scores_shape_usize: Vec<usize> = scores_shape.iter().map(|&x| x as usize).collect();
            let coords_shape_usize: Vec<usize> = coords_shape.iter().map(|&x| x as usize).collect();

            let scores = ArrayView::from_shape(scores_shape_usize, scores_data)?
                .into_dimensionality::<Ix3>()?;
            let coords = ArrayView::from_shape(coords_shape_usize, coords_data)?
                .into_dimensionality::<Ix3>()?;

            // 2. Find the best detection (Argmax over 2944 candidates)
            let mut max_score = 0.0;
            let mut best_idx = 0;

            for i in 0..2944 {
                let score = scores[[0, i, 0]];
                if score > max_score {
                    max_score = score;
                    best_idx = i;
                }
            }

            // 3. Threshold check: If the score is too low, no hand is visible
            if max_score < 0.5 {
                return Ok(None);
            }

            // 4. Extract coordinates for the best index
            // Typically: [ymin, xmin, ymax, xmax, kp1_y, kp1_x, ...]
            let y_center = coords[[0, best_idx, 0]];
            let x_center = coords[[0, best_idx, 1]];

            // For now, let's just return a gesture if we find a hand with high confidence

            Ok(Some(DetailedGesture {
                gesture: Gesture::Pointer,
                x: x_center,
                y: y_center,
                score: max_score,
            }))
        }
    }
}
