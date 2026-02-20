pub mod webcam {

    use image::{ImageBuffer, Rgb};
    use nokhwa::pixel_format::RgbFormat;
    use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
    use nokhwa::{Camera, NokhwaError};

    pub fn setup() -> Result<Camera, nokhwa::NokhwaError> {
        // Setup Camera //
        let index = CameraIndex::Index(0);
        let requested =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        println!("Opening camera...");

        // Return
        Camera::new(index, requested)
    }

    pub fn capture_and_decode_frame(
        camera: &mut Camera,
    ) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, NokhwaError> {
        // Capture frame
        let frame = camera.frame()?;
        // Decode frame as image
        let decoded = frame.decode_image::<RgbFormat>()?;
        // Return
        Ok(decoded)
    }
}
