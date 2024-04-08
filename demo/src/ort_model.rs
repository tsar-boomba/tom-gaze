use std::path::Path;

use color_eyre::eyre::Result;
use ndarray as nd;
use ort::{ExecutionProviderDispatch, Session, Value};

pub struct OrtModel {
    model: Session,
}

impl OrtModel {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let model = ort::SessionBuilder::new()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_parallel_execution(true)?
            .with_execution_providers([ExecutionProviderDispatch::CoreML(Default::default())])?
            .commit_from_file(path)?;

        println!("inputs: {:?}", model.inputs);

        Ok(Self { model })
    }

    pub fn predict<'a>(&self, input: nd::Array4<f32>) -> Result<(f32, f32)> {
        let out = self.model.run(ort::inputs![Value::from_array(input)?]?)?;

        let result = out[0].try_extract_tensor::<f32>()?;
        let view = result.view();
        let rows = view.rows().into_iter().next().unwrap();
        let pitch = rows[1];
        let yaw = rows[0];

        let vector = -nd::Array1::from_iter([
            f32::cos(pitch) * f32::sin(yaw),
            f32::sin(pitch),
            f32::cos(pitch) * f32::cos(yaw),
        ]);

        let pitch = f32::asin(-vector[1]);
        let yaw = f32::atan2(-vector[0], -vector[2]);
        Ok((pitch, yaw))
    }
}
