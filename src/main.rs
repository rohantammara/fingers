use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;
use minifb::{Window, WindowOptions, Key};
use std::time::Duration;

fn main() -> anyhow::Result<()> {
    // Setup Camera //
    let index = CameraIndex::Index(0);
    let requested = RequestedFormat::new::<RgbFormat>(
        RequestedFormatType::AbsoluteHighestFrameRate
    );
    println!("Opening camera...");
    let mut camera = Camera::new(index, requested)?;
    camera.open_stream()?;

    // Get the camera's resolution to ensure window and frame sizes match.
    let resolution = camera.resolution();
    println!("Camera resolution: {}x{}", resolution.width(), resolution.height());
    let width = resolution.width() as usize;
    let height = resolution.height() as usize;

    // Setup Window (The "Screen") //
    let mut window = Window::new(
        "Fingers (Press ESC to exit)",
        width,
        height,
        WindowOptions::default(),
    )?;
    
    // Limit to ~60 fps to reduce CPU usage and potential instability
    window.limit_update_rate(Some(Duration::from_micros(16600)));

    // The Loop
    while window.is_open() && !window.is_key_down(Key::Escape) {
        
        // A. Capture Frame
        let frame = match camera.frame() {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Frame capture error: {}", e);
                continue;
            }
        };
        
        // Pixel Conversion //
        // The camera gives us a long list of u8 bytes: [R, G, B, R, G, B...]
        // The window wants u32 integers: [00RGB, 00RGB...]
        // We must map them.

        let decoded = match frame.decode_image::<RgbFormat>() {
            Ok(image) => image,
            Err(e) => {
                eprintln!("Frame decode error: {}", e);
                continue;
            }
        };
        let raw_data = decoded.as_raw();

        // Sanity check buffer size
        if raw_data.len() != width * height * 3 {
            eprintln!("Buffer size mismatch: Expected {}, got {}", width * height * 3, raw_data.len());
            continue;
        }
        
        // This 'map' creates a new vector of u32 pixels
        let buffer_u32: Vec<u32> = raw_data
            .chunks_exact(3) // Take 3 bytes at a time (R, G, B)
            .map(|chunk| {
                let r = chunk[0] as u32;
                let g = chunk[1] as u32;
                let b = chunk[2] as u32;
                // Combine them into one u32 number (00 | R | G | B)
                (r << 16) | (g << 8) | b
            })
            .collect();

        // Draw to Window
        // This is roughly equivalent to 'cv2.imshow()' + 'cv2.waitKey()'
        window.update_with_buffer(&buffer_u32, width, height)?;
    }

    Ok(())
}