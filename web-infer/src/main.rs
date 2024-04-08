use std::sync::OnceLock;

use ultraface::UltrafaceModel;

static ULTRAFACE_MODEL: &'static [u8] = include_bytes!("../../assets/ultraface-RFB-320-quant.onnx");
static FACE_DETECTOR: OnceLock<UltrafaceModel<'static>> = OnceLock::new();

const WIDTH: usize = 320;
const HEIGHT: usize = 240;

static mut TRANSFORM_BUFFER: [f32; WIDTH * HEIGHT * 3] = [0.0; WIDTH * HEIGHT * 3];

#[no_mangle]
pub extern "C" fn infer(image: *mut u8, len: usize) {
    let image = unsafe { std::slice::from_raw_parts_mut(image, WIDTH * HEIGHT * 4) };

    // Copy only r, g, b channels to new buffer
    // [r, g, b, a, r, g, b, a, r, g, b, a]
    // [r, g, b, r, g, b, r, g, b]
    for (byte_num, byte_start) in (0..image.len()).step_by(4).enumerate() {
        unsafe {
            TRANSFORM_BUFFER[byte_num * 3] = image[byte_start] as f32;
            TRANSFORM_BUFFER[(byte_num * 3) + 1] = image[byte_start + 1] as f32;
            TRANSFORM_BUFFER[(byte_num * 3) + 2] = image[byte_start + 2] as f32;
        }
    }

    let faces = FACE_DETECTOR
        .get()
        .unwrap()
        .detect(unsafe { &TRANSFORM_BUFFER })
        .unwrap();

    println!("{faces:?}");
}

/// Initialize everything needed for inference
fn main() {
    println!("Hello from Rust!");

    FACE_DETECTOR
        .set(
            ultraface::UltrafaceModel::new(
                ultraface::UltrafaceVariant::W320H240Quantized,
                [],
                ULTRAFACE_MODEL,
                0.5,
                0.72,
            )
            .unwrap(),
        )
        .unwrap();

    println!("Initialized WASM.");
}
