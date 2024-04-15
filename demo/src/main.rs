mod ort_model;
mod utraface;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::time::Instant;

use color_eyre::Result;
use ndarray as nd;
use nd::Axis;
use opencv::{
    core::{Point, Rect, Scalar, CV_8U},
    highgui,
    imgproc::{self, COLOR_BGR2RGB, INTER_NEAREST, LINE_AA},
    prelude::*,
    videoio,
};
use ort_model::OrtModel;
use utraface::{UltrafaceModel, UltrafaceVariant};

const MODEL: &'static str = "./assets/eth-xgaze_resnet18";
const CAM_WIDTH: f32 = 1920.0;
const CAM_HEIGHT: f32 = 1080.0;

fn main() -> Result<()> {
    tracing_subscriber::fmt().init();

    // Init neural network
    let ort_model = OrtModel::new(format!("{}.onnx", MODEL))?;

    // Init face detection
    let ultraface_variant = UltrafaceVariant::W320H240;
    let face_detector = UltrafaceModel::new(ultraface_variant, 0.5, 0.68)?;

    let mut camera = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    let mut frame = Mat::default();

    let mut frame_num = 0u64;
    loop {
        println!("\n--Start frame {frame_num}--");
        frame_num += 1;
        camera.read(&mut frame)?;
        let frame_start = Instant::now();
        let start = Instant::now();
        let faces = face_detector.detect(&frame)?;
        println!(
            "Face detect ({} faces): {}ms",
            faces.len(),
            (Instant::now() - start).as_millis()
        );

        for (rect, _confidence) in faces {
            println!("{rect:#?}");
            let rect = Rect {
                x: (rect.x * CAM_WIDTH) as i32,
                y: (rect.y * CAM_HEIGHT) as i32,
                width: (rect.width * CAM_WIDTH) as i32,
                height: (rect.height * CAM_HEIGHT) as i32,
            };

            let face_mat = match Mat::roi(&frame, rect) {
                Ok(face_mat) => face_mat.clone(),
                // Error usually means bounding box went outside of frame
                Err(_) => continue,
            };

            let resized_face = face_to_224_nd(face_mat)?;
            let resized_face = resized_face.insert_axis(Axis(0));

            let start = Instant::now();
            let (pitch, yaw) = ort_model.predict(resized_face)?;
            println!("Predict (Ort): {}ms", (Instant::now() - start).as_millis());

            let start = Instant::now();
            // Make the vector a little longer
            let length = rect.width;
            let center = (
                rect.x as f32 + (rect.width as f32 / 2.0),
                rect.y as f32 + (rect.height as f32 / 2.0),
            );

            let dx = -length as f32 * f32::sin(pitch) * f32::cos(yaw);
            let dy = -length as f32 * f32::sin(yaw);

            imgproc::arrowed_line(
                &mut frame,
                Point::new(center.0.round() as i32, center.1.round() as i32),
                Point::new(
                    (center.0 + dx).round() as i32,
                    (center.1 + dy).round() as i32,
                ),
                // BGR
                Scalar::new(255., 255., 0., 0.),
                3,
                LINE_AA,
                0,
                0.18,
            )?;

            imgproc::rectangle(
                &mut frame,
                rect,
                Scalar::new(0., 255., 0f64, 0.),
                2,
                imgproc::LINE_8,
                0,
            )?;
            println!("Draw to image: {}µs", (Instant::now() - start).as_micros());
        }

        println!(
            "--Total Processing Time: {}ms--\n",
            (Instant::now() - frame_start).as_millis()
        );
        highgui::imshow("frame", &frame)?;
        highgui::poll_key()?;
    }
}

pub fn face_to_224_nd(mut face_mat: Mat) -> Result<nd::Array3<f32>> {
    let channels = face_mat.channels();

    assert!(face_mat.depth() == CV_8U);
    assert!(channels == 3);
    assert!(face_mat.is_continuous());

    let dst = &mut Mat::zeros(224, 224, CV_8U)?.to_mat()?;
    opencv::imgproc::cvt_color(&face_mat.clone(), &mut face_mat, COLOR_BGR2RGB, 0)?;
    opencv::imgproc::resize(&face_mat, dst, dst.size()?, 0.0, 0.0, INTER_NEAREST)?;

    let data_len = (224 * 224 * 3) as usize;
    let mat_data = unsafe { std::slice::from_raw_parts(dst.ptr(0)?, data_len) };
    let mut vec = Vec::with_capacity(mat_data.len() * 4);

    for byte in mat_data {
        vec.push(*byte as f32);
    }

    let array = nd::Array3::from_shape_vec((224, 224, 3), vec)?.permuted_axes([2, 0, 1]);
    let mean = nd::Array1::from_iter([0.485f32, 0.456, 0.406]).into_shape((3, 1, 1))?;
    let std = nd::Array1::from_iter([0.229f32, 0.224, 0.225]).into_shape((3, 1, 1))?;
    Ok(((array / 255f32) - mean.broadcast((3, 224, 224)).unwrap())
        / std.broadcast((3, 224, 224)).unwrap())
}
