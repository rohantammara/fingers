pub mod webcam {

    use nokhwa::pixel_format::RgbFormat;
    use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
    use nokhwa::{Camera, NokhwaError};
    use image::{ImageBuffer, Rgb};

    pub fn setup() -> Result<Camera, nokhwa::NokhwaError> {
        
        // Setup Camera //
        let index = CameraIndex::Index(0);
        let requested = RequestedFormat::new::<RgbFormat>(
            RequestedFormatType::AbsoluteHighestFrameRate
        );
        println!("Opening camera...");

        // Return
        Camera::new(index, requested)
    }

    pub fn start_stream(camera: &mut Camera) -> Result<(usize, usize), NokhwaError> {
    
        // Start camera
        camera.open_stream()?;

        // Get the camera's resolution to ensure window and frame sizes match.
        let resolution = camera.resolution();
        println!("Camera resolution: {}x{}", resolution.width(), resolution.height());
        let width = resolution.width() as usize;
        let height = resolution.height() as usize;
        
        // Return
        Ok((width, height))
    }

    pub fn capture_and_decode_frame(camera: &mut Camera) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, NokhwaError>{
        
        // Capture frame
        let frame = camera.frame()?;
        // Decode frame as image
        let decoded = frame.decode_image::<RgbFormat>()?;
        // Return
        Ok(decoded)
    }
}