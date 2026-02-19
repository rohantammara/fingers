pub mod input_device {

    use anyhow::Error;
    use enigo::{Enigo, Settings, Mouse};

    pub fn create() -> Result<Enigo, Error> {

        // Setup Input Controller (Enigo)
        let enigo_controller = Enigo::new(&Settings::default()).unwrap();

        // Return
        Ok(enigo_controller)
    }
    
}