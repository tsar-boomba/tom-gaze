//! Neural network module with model struct, pre- and post-process functions.
//!
use color_eyre::eyre::Result;
use cv_convert::prelude::*;
use nd::{s, Axis};
use ndarray as nd;
use opencv::{
    core::{Point2f, Rect_, CV_8U},
    imgproc::INTER_NEAREST,
    prelude::*,
};
use ort::{ExecutionProviderDispatch, Session, SessionOutputs, Value};

/// Positive additive constant to avoid divide-by-zero.
const EPS: f32 = 1.0e-7;
type Rect = Rect_<f32>;

// Links to the Ultraface model files
const ULTRAFACE_640: &str = "./assets/ultraface-RFB-640.onnx";
const ULTRAFACE_320: &str = "./assets/ultraface-RFB-320.onnx";

/// Supported variants of the Ultraface model.
pub enum UltrafaceVariant {
    W640H480,
    W320H240,
    W320H240Quantized,
}

impl UltrafaceVariant {
    /// Get width and height from the UltrafaceVariant.
    pub fn width_height(&self) -> (u32, u32) {
        match self {
            UltrafaceVariant::W640H480 => (640, 480),
            UltrafaceVariant::W320H240 | UltrafaceVariant::W320H240Quantized => (320, 240),
        }
    }

    pub fn path(&self) -> &str {
        match self {
            UltrafaceVariant::W640H480 => ULTRAFACE_640,
            UltrafaceVariant::W320H240 => ULTRAFACE_320,
            UltrafaceVariant::W320H240Quantized => "ultraface-RFB-320-quant.onnx",
        }
    }
}

/// Loaded Ultraface model, ready for inference with post-processing thresholds.
pub struct UltrafaceModel {
    model: Session,
    width: u32,
    height: u32,
    max_iou: f32,
    min_confidence: f32,
}

impl UltrafaceModel {
    /// Load and prepare an Ultraface model for inference.
    pub fn new(variant: UltrafaceVariant, max_iou: f32, min_confidence: f32) -> Result<Self> {
        let (width, height) = variant.width_height();
        let model = ort::SessionBuilder::new()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_inter_threads(3)?
            .with_parallel_execution(true)?
            .with_execution_providers([ExecutionProviderDispatch::CoreML(Default::default())])?
            .commit_from_file(variant.path())?;
        println!("Initialized Ultraface model");

        Ok(Self {
            model,
            width,
            height,
            max_iou,
            min_confidence,
        })
    }

    pub fn detect(&self, input: &Mat) -> Result<Vec<(Rect, f32)>> {
        let valid_input = self.preproc(input);
        let raw_nn_out = self
            .model
            .run(ort::inputs![Value::from_array(valid_input)?]?)?;
        let selected_bboxes = self.postproc(&raw_nn_out)?;

        Ok(selected_bboxes)
    }

    /// Pre-process an image to be used as inference input.
    fn preproc(&self, input: &Mat) -> nd::Array4<f32> {
        let dst = &mut Mat::zeros(self.height as i32, self.width as i32, CV_8U)
            .unwrap()
            .to_mat()
            .unwrap();
        opencv::imgproc::resize(&input, dst, dst.size().unwrap(), 0.0, 0.0, INTER_NEAREST).unwrap();
        let array: nd::Array3<f32> = nd::Array3::<u8>::try_from_cv(&*dst)
            .unwrap()
            .map(|b| *b as f32)
            .permuted_axes([2, 0, 1]);

        println!("{:?}", array.shape());

        let mean = nd::Array1::from_iter([0.485f32, 0.456, 0.406])
            .into_shape((3, 1, 1))
            .unwrap();
        let std = nd::Array1::from_iter([0.229f32, 0.224, 0.225])
            .into_shape((3, 1, 1))
            .unwrap();
        let img_shape = array.dim();

        (((array / 255f32) - mean.broadcast(img_shape).unwrap())
            / std.broadcast(img_shape).unwrap())
        .insert_axis(Axis(0))
    }

    /// Post-process raw inference output to selected bounding boxes.
    ///
    /// The raw inference output `raw_nn_out` consist of two tensors:
    /// - `raw_nn_out[0]` is a `1xKx2` tensor of bounding box confidences. The confidences for
    ///   having a face in a bounding box are given in the second column at `[:,:,1]`.
    /// - `raw_nn_out[1]` is a `1xKx4` tensor of bounding box candidate border points. Every
    ///   candidate bounding box consists of the **relative** coordinates
    ///   `[x_top_left, y_top_left, x_bottom_right, y_bottom_right]`. They can be multiplied with
    ///   the `width` and `height` of the original image to obtain the bounding box coordinates for
    ///   the real frame.
    ///
    /// The output is a vector of bounding boxes with confidence scores in descending order of
    /// certainty. The bounding boxes are defined by their **relative** coordinates.
    fn postproc(&self, raw_nn_out: &SessionOutputs<'_>) -> Result<Vec<(Rect, f32)>> {
        // Extract confidences
        let confidences = raw_nn_out[0]
            .try_extract_tensor::<f32>()?
            .view()
            .to_owned()
            .slice(s![0, .., 1])
            .to_vec();

        // Extract relative coordinates of bounding boxes
        let bboxes: Vec<f32> = raw_nn_out[1]
            .try_extract_tensor::<f32>()?
            .view()
            .iter()
            .cloned()
            .collect();
        let bboxes: Vec<Rect> = bboxes
            .chunks(4)
            .map(|bbox| {
                Rect::from_points(
                    Point2f::new(bbox[0], bbox[1]),
                    Point2f::new(bbox[2], bbox[3]),
                )
            })
            .collect();

        // Fuse bounding boxes with confidence scores
        // Filter out bounding boxes with a confidence score below the threshold
        let mut bboxes_with_confidences: Vec<_> = bboxes
            .iter()
            .zip(confidences.iter())
            .filter_map(|(bbox, confidence)| match confidence {
                x if *x > self.min_confidence => Some((bbox, confidence)),
                _ => None,
            })
            .collect();

        // Sort pairs of bounding boxes with confidence scores by **ascending** confidences to allow
        // cheap removal of the top candidates from the back
        bboxes_with_confidences.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());

        // Run non-maximum suppression on the sorted vector of bounding boxes with confidences
        let selected_bboxes = non_maximum_suppression(bboxes_with_confidences, self.max_iou);

        Ok(selected_bboxes)
    }
}

/// Run non-maximum-suppression on candidate bounding boxes.
///
/// The pairs of bounding boxes with confidences have to be sorted in **ascending** order of
/// confidence because we want to `pop()` the most confident elements from the back.
///
/// Start with the most confident bounding box and iterate over all other bounding boxes in the
/// order of decreasing confidence. Grow the vector of selected bounding boxes by adding only those
/// candidates which do not have a IoU scores above `max_iou` with already chosen bounding boxes.
/// This iterates over all bounding boxes in `sorted_bboxes_with_confidences`. Any candidates with
/// scores generally too low to be considered should be filtered out before.
fn non_maximum_suppression(
    mut sorted_bboxes_with_confidences: Vec<(&Rect, &f32)>,
    max_iou: f32,
) -> Vec<(Rect, f32)> {
    let mut selected = vec![];
    'candidates: loop {
        // Get next most confident bbox from the back of ascending-sorted vector.
        // All boxes fulfill the minimum confidence criterium.
        match sorted_bboxes_with_confidences.pop() {
            Some((bbox, confidence)) => {
                // Check for overlap with any of the selected bboxes
                for (selected_bbox, _) in selected.iter() {
                    match iou(bbox, selected_bbox) {
                        x if x > max_iou => continue 'candidates,
                        _ => (),
                    }
                }

                // bbox has no large overlap with any of the selected ones, add it
                selected.push((*bbox, *confidence))
            }
            None => break 'candidates,
        }
    }

    selected
}

/// Calculate the intersection-over-union metric for two bounding boxes.
fn iou(bbox_a: &Rect, bbox_b: &Rect) -> f32 {
    // Calculate corner points of overlap box
    // If the boxes do not overlap, the corner-points will be ill defined, i.e. the top left
    // corner point will be below and to the right of the bottom right corner point. In this case,
    // the area will be zero.
    let overlap_box = Rect::from_points(
        Point2f::new(f32::max(bbox_a.x, bbox_b.x), f32::max(bbox_a.y, bbox_b.y)),
        Point2f::new(
            f32::min(bbox_a.x + bbox_a.width, bbox_b.x + bbox_b.width),
            f32::min(bbox_a.y + bbox_a.height, bbox_b.y + bbox_b.height),
        ),
    );

    let overlap_area = bbox_area(&overlap_box);

    // Avoid division-by-zero with `EPS`
    overlap_area / (bbox_area(bbox_a) + bbox_area(bbox_b) - overlap_area + EPS)
}

/// Calculate the area enclosed by a bounding box.
///
/// The bounding box is passed as four-element array defining two points:
/// `[x_top_left, y_top_left, x_bottom_right, y_bottom_right]`
/// If the bounding box is ill-defined by having the bottom-right point above/to the left of the
/// top-left point, the area is zero.
fn bbox_area(bbox: &Rect) -> f32 {
    if bbox.width < 0.0 || bbox.height < 0.0 {
        // bbox is empty/undefined since the bottom-right corner is above the top left corner
        return 0.0;
    }

    bbox.area()
}
