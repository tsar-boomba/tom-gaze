import { useEffect, useRef } from 'react';
import './App.css';
import { InferenceSession } from 'onnxruntime-web';
import { inferFace, inferGaze } from './infer';
import { OUTPUT_HEIGHT, OUTPUT_WIDTH, CAMERA_WIDTH, CAMERA_HEIGHT, MODEL_WIDTH, MODEL_HEIGHT, DEBUG } from './consts';

function App() {
	useEffect(() => {
		setTimeout(async () => {
			for (const device of await navigator.mediaDevices.enumerateDevices()) {
				if (device.kind === 'videoinput') {
					const stream = await navigator.mediaDevices.getUserMedia({
						video: true,
						audio: false,
						preferCurrentTab: true,
					});

					console.log({ device, stream, tracks: stream.getTracks() });

					const video = document.createElement('video');
					video.srcObject = stream;
					video.width = MODEL_WIDTH;
					video.height = MODEL_HEIGHT;

					try {
						await video.play();
					} finally {
						// asd;lkjasdfjkl;
					}

					const imgCanvas = new OffscreenCanvas(MODEL_WIDTH, MODEL_HEIGHT);
					const imgContext = imgCanvas.getContext('2d', {
						willReadFrequently: true,
						alpha: false,
					})!;

					const resultCanvas = document.createElement('canvas');
					resultCanvas.width = OUTPUT_WIDTH;
					resultCanvas.height = OUTPUT_HEIGHT;
					resultCanvas.style.margin = 'auto';
					document.body.appendChild(resultCanvas);
					const resultContext = resultCanvas.getContext('2d', {
						alpha: false,
					})!;
					resultContext.lineWidth = 2;

					const aoiCanvas = new OffscreenCanvas(224, 224);
					const aoiCtx = aoiCanvas.getContext('2d', {
						alpha: false,
						willReadFrequently: true,
					})!;

					// ultraface-RFB-320-quant.onnx
					const faceRes = await fetch('/ultraface-RFB-320-sim.onnx');
					const faceSession = await InferenceSession.create(await faceRes.arrayBuffer(), {
						executionMode: 'sequential',
						graphOptimizationLevel: 'all',
						extra: {
							optimization: {
								enable_gelu_approximation: "1"
							}
						},
						executionProviders: [
							{
								name: 'webgl',
							}
						]
					});

					const gazeRes = await fetch('/eth-xgaze_resnet18-sim.onnx');
					const gazeSession = await InferenceSession.create(await gazeRes.arrayBuffer(), {
						executionMode: 'sequential',
						graphOptimizationLevel: 'all',
						extra: {
							optimization: {
								enable_gelu_approximation: "1"
							}
						},
						executionProviders: [{ name: 'wasm' }]
					});

					try {
						// eslint-disable-next-line no-constant-condition
						while (1) {
							const start = Date.now();
							DEBUG && console.log('----- Frame Start -----');
							imgContext.drawImage(video, 0, 0, MODEL_WIDTH, MODEL_HEIGHT);
							const img = imgContext.getImageData(0, 0, MODEL_WIDTH, MODEL_HEIGHT);
							const boxes = await inferFace(img, faceSession);

							resultContext.putImageData(img, 0, 0);
							for (const [box] of boxes) {
								const x = Math.round(box.x * OUTPUT_WIDTH);
								const y = Math.round(box.y * OUTPUT_HEIGHT);
								const width = Math.round(box.width * OUTPUT_WIDTH);
								const height = Math.round(box.height * OUTPUT_HEIGHT);

								resultContext.strokeStyle = '#00FF00FF';
								resultContext.strokeRect(x, y, width, height);

								aoiCtx.drawImage(imgCanvas, x, y, width, height, 0, 0, 224, 224);

								// const testc = document.createElement('canvas');
								// testc.width = 224;
								// testc.height = 224;
								// testc.getContext('2d')!.putImageData(aoiCtx.getImageData(0, 0, 224, 224), 0, 0);
								// console.log(testc.toDataURL());

								const { pitch, yaw } = await inferGaze(aoiCtx.getImageData(0, 0, 224, 224), gazeSession);
								const centerX = x + (width / 2);
								const centerY = y + (height / 2);

								const dx = -width * Math.sin(pitch) * Math.cos(yaw);
								const dy = -width * Math.sin(yaw);
								resultContext.strokeStyle = '#00FFFFFF';
								resultContext.beginPath();
								resultContext.moveTo(centerX, centerY);
								resultContext.lineTo(centerX + dx, centerY + dy);
								resultContext.stroke();
							}

							const totalTime = Date.now() - start;
							DEBUG && console.log('Frame total (ms):', totalTime);
							DEBUG && console.log('----- Frame End -----');
							await new Promise((res) => setTimeout(res, 13));
						}
					} finally {
						await faceSession.release();
					}
				}
			}
		}, 0);
	}, []);

	return <></>;
}

export default App;
