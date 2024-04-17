import { InferenceSession, Tensor } from "onnxruntime-web";
import { DEBUG, MODEL_HEIGHT, MODEL_WIDTH } from "./consts";

type Rect = { x: number; y: number; width: number; height: number };

const DIMS = [3, MODEL_HEIGHT, MODEL_WIDTH];
const GAZE_DIMS = [3, 224, 224];
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const MIN_CONFIDENCE = 0.72;
const MAX_IOU = 0.5;

const normalize = ({ width, height, data }: ImageData) => {
    const dst = new Float32Array(width * height * 3);
    const step = width * height;
    for(let y = 0; y < height; y++) {
		for(let x = 0; x < width; x++) {
			const [di, si] = [y * width + x, (y * width + x) * 4];
			// 各チャンネルで平均を引いて標準偏差で割る(標準化)
			// さらに RGBARGBARGBA... から RRR...GGG...BBB... の順にデータを詰め替え
			dst[di] = ((data[si + 0] / 255) - MEAN[0]) / STD[0];
			dst[di + step] = ((data[si + 1] / 255) - MEAN[1]) / STD[1];
			dst[di + step * 2] = ((data[si + 2] / 255) - MEAN[2]) / STD[2];
		}
    }
    return dst;
}

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

export const inferFace = async (image: ImageData, session: InferenceSession) => {
	if (image.width !== MODEL_WIDTH && image.height !== MODEL_HEIGHT)
		throw new Error(`Image must be ${MODEL_WIDTH} x ${MODEL_HEIGHT}`);

	DEBUG && console.log('--- Face Start ---');
	const normalizeStart = Date.now();
	const normalized = normalize(image);
	DEBUG && console.log('Preprocessed in (ms):', Date.now() - normalizeStart);

	const inferStart = Date.now();
	const imgTensor = new Tensor('float32', normalized, [1, ...DIMS]);

	const { boxes, scores } = await session.run({ input: imgTensor });
	DEBUG && console.log('Inferred in (ms):', Date.now() - inferStart);
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

	DEBUG && console.log('Postprocessed in (ms):', Date.now() - postStart);
	DEBUG && console.log('--- Face End ---');
	return seleceted;
}

export const inferGaze = async (face: ImageData, session: InferenceSession) => {
	if (face.width !== 224 && face.height !== 224)
		throw new Error(`Image must be 224 x 224, but was ${face.width} x ${face.height}`);

	DEBUG && console.log('--- Gaze Start ---');
	const normalizeStart = Date.now();
	const normalized = normalize(face);
	DEBUG && console.log('Preprocessed in (ms):', Date.now() - normalizeStart);

	const inferStart = Date.now();
	const imgTensor = new Tensor('float32', normalized, [1, ...GAZE_DIMS]);

	const rawOut = await session.run({ 'input.1': imgTensor });
	const data = rawOut['191'].data;
	const pitch = data[1] as number;
	const yaw = data[0] as number;
	// console.log({ pitsh: pitchRaw, yaw: yawRaw });

	const vec = [
		-(Math.cos(pitch) * Math.sin(yaw)),
		-(Math.sin(pitch)),
		-(Math.cos(pitch) * Math.cos(yaw)),
    ];

	// const pitch = Math.asin(-vec[1]);
	// const yaw = Math.atan2(-vec[0], -vec[2]);

	DEBUG && console.log('Inferred in (ms):', Date.now() - inferStart);
	DEBUG && console.log('--- Gaze End ---');

	return { pitch, yaw, vec };
}
