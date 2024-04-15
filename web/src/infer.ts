import { InferenceSession, Tensor } from "onnxruntime-web";
import { CAMERA_HEIGHT, CAMERA_WIDTH, MODEL_HEIGHT, MODEL_WIDTH } from "./consts";

type Rect = { x: number; y: number; width: number; height: number };

const DIMS = [3, MODEL_HEIGHT, MODEL_WIDTH];
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const MIN_CONFIDENCE = 0.72;
const MAX_IOU = 0.5;

const rectFromPoints = (pt1x: number, pt1y: number, pt2x: number, pt2y: number) => {
	const x = Math.min(pt1x, pt2x);
    const y = Math.min(pt1y, pt2y);

	return {
		x,
		y,
		width: Math.max(pt1x, pt2x) - x,
		height: Math.max(pt1y, pt2y) - y,
	}
};

const area = (r: Rect) => {
	if (r.width < 0 || r.height < 0) return 0;
	return r.width * r.height
};

const iou = (r1: Rect, r2: Rect) => {
	// let overlap_box = Rect::from_points(
    //     Point::new(f32::max(bbox_a.x, bbox_b.x), f32::max(bbox_a.y, bbox_b.y)),
    //     Point::new(
    //         f32::min(bbox_a.x + bbox_a.width, bbox_b.x + bbox_b.width),
    //         f32::min(bbox_a.y + bbox_a.height, bbox_b.y + bbox_b.height),
    //     ),
    // );

    // let overlap_area = bbox_area(&overlap_box);

    // // Avoid division-by-zero with `EPS`
    // overlap_area / (bbox_area(bbox_a) + bbox_area(bbox_b) - overlap_area + EPS)

	const overlapBox = rectFromPoints(
		Math.max(r1.x, r2.x),
		Math.max(r1.y, r2.y),
		Math.min(r1.x + r1.width, r2.x + r2.width),
		Math.min(r1.y + r1.height, r2.y + r2.height)
	);

	const overlapArea = area(overlapBox);

	return overlapArea / (area(r1) + area(r2) - overlapArea + 1e-7)
}

const nonMaximumSuppression = (boxes: [Rect, number][]) => {
	const selected: [Rect, number][] = [];
	candidates: while (boxes.length) {
		const [box, confidence] = boxes.pop()!;

		for (const [select] of selected) {
			if (iou(box, select) > MAX_IOU) {
				continue candidates;
			}
		}

		selected.push([box, confidence]);
	}

	return selected;
}

export const infer = async (image: ImageData, session: InferenceSession) => {
	const inferStart = Date.now();
	console.log(image);
	const imgTensor = await Tensor.fromImage(image, {
		resizedWidth: MODEL_WIDTH,
		resizedHeight: MODEL_HEIGHT,
		norm: {
			mean: MEAN,
			bias: STD,
		}
	});

	const { boxes, scores } = await session.run({ input: imgTensor.reshape([1, ...DIMS]) });
	console.log('Inferred in (ms):', Date.now() - inferStart);
	const postStart = Date.now();
	
	const rawScores = scores.data as Float32Array;
	const rawBoxes = boxes.data as Float32Array;
	const bboxes: [Rect, number][] = [];

	let boxNum = 0;
	for (let i = 0; i < rawBoxes.length; i += 4) {
		const scoreIdx = (boxNum * 2) + 1;
		if (rawScores[scoreIdx] > MIN_CONFIDENCE) {
			bboxes.push([rectFromPoints(rawBoxes[i], rawBoxes[i + 1], rawBoxes[i + 2], rawBoxes[i + 3]), rawScores[scoreIdx]]);
		}
		boxNum++;
	}

	bboxes.sort(([, confA], [, confB]) => confA - confB);
	
	const seleceted = nonMaximumSuppression(bboxes);

	console.log('Postprocessed in (ms):', Date.now() - postStart);
	return seleceted;
}
