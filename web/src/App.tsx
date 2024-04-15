import { useEffect, useRef } from 'react';
import './App.css';
import { InferenceSession } from 'onnxruntime-web';
import { infer } from './infer';
import { OUTPUT_HEIGHT, OUTPUT_WIDTH, CAMERA_WIDTH, CAMERA_HEIGHT } from './consts';

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
					video.width = CAMERA_WIDTH;
					video.height = CAMERA_HEIGHT;

					try {
						await video.play();
					} finally {
						// asd;lkjasdfjkl;
					}

					const imgCanvas = new OffscreenCanvas(CAMERA_WIDTH, CAMERA_HEIGHT);
					const imgContext = imgCanvas.getContext('2d', {
						willReadFrequently: true,
						alpha: false,
					})!;

					const resultCanvas = document.createElement('canvas');
					resultCanvas.width = OUTPUT_WIDTH;
					resultCanvas.height = OUTPUT_HEIGHT;
					document.body.appendChild(resultCanvas);
					const resultContext = resultCanvas.getContext('2d', {
						alpha: false,
					})!;
					resultContext.strokeStyle = '#00FF00FF';
					resultContext.lineWidth = 2;

					// ultraface-RFB-320-quant.onnx
					const modelRes = await fetch('/ultraface-RFB-320-sim.onnx');
					const session = await InferenceSession.create(await modelRes.arrayBuffer(), {
						executionMode: 'sequential',
						graphOptimizationLevel: 'all',
						extra: {
							optimization: {
								enable_gelu_approximation: "1"
							}
						}
					});

					try {
						// eslint-disable-next-line no-constant-condition
						while (1) {
							imgContext.drawImage(video, 0, 0, CAMERA_WIDTH, CAMERA_HEIGHT);
							const img = imgContext.getImageData(0, 0, CAMERA_WIDTH, CAMERA_HEIGHT);
							const boxes = await infer(img, session);

							resultContext.putImageData(img, 0, 0);
							for (const [box] of boxes) {
								const x = box.x * OUTPUT_WIDTH;
								const y = box.y * OUTPUT_HEIGHT;
								const width = box.width * OUTPUT_WIDTH;
								const height = box.height * OUTPUT_HEIGHT;

								//console.log(box);

								resultContext.strokeRect(x, y, width, height);
							}

							await new Promise((res) => setTimeout(res, 3000));
						}
					} finally {
						await session.release();
					}
				}
			}
		}, 0);
	}, []);

	const input = useRef<HTMLInputElement>(null);

	return (
		<>
			{/* <input type='file' ref={input} onChange={async (e) => {
				const file = e.target.files![0];
				const reader = new FileReader();
				const dataUrl = await new Promise<string>((res) => {
					reader.onload = (e) => {
						res(e.target?.result as string);
					}
					reader.readAsDataURL(file);
				});
				
				const img = new Image();
				img.src = dataUrl;
				document.body.appendChild(img);
				img.onload = () => {
					console.log({ w: img.width, h: img.height })
				}
			}} /> */}
		</>
	);
}

export default App;
