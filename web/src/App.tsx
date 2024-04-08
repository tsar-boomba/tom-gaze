import { useEffect, useState } from 'react';
import reactLogo from './assets/react.svg';
import viteLogo from '/vite.svg';
import './App.css';

const WIDTH = 320;
const HEIGHT = 240;
const IMG_BUFFER_LEN = WIDTH * HEIGHT * 4;

function App() {
	const [count, setCount] = useState(0);

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
					video.autoplay = true;
					document.body.appendChild(video);

					await video.play();

					const canvas = new OffscreenCanvas(WIDTH, HEIGHT);
					const context = canvas.getContext('2d', {
						willReadFrequently: true,
						alpha: false,
					})!;

					await new Promise((res) => setTimeout(res, 1000));
					const ptr = _malloc(IMG_BUFFER_LEN);
					const dataOnHeap = new Uint8Array(
						Module.HEAPU8.buffer,
						ptr,
						IMG_BUFFER_LEN
					);

					// eslint-disable-next-line no-constant-condition
					while (1) {
						context.drawImage(video, 0, 0, WIDTH, HEIGHT);
						const img = context.getImageData(0, 0, WIDTH, HEIGHT);
						dataOnHeap.set(img.data);
						const start = Date.now();
						Module._infer(ptr, IMG_BUFFER_LEN);
						console.log('Frame process time (ms):', Date.now() - start);
						// Block this task temporarily so that other things can happen
						await new Promise((res) => setTimeout(res, 200));
					}
				}
			}
		}, 0);
	}, []);

	return (
		<>
			<div>
				<a href='https://vitejs.dev' target='_blank'>
					<img src={viteLogo} className='logo' alt='Vite logo' />
				</a>
				<a href='https://react.dev' target='_blank'>
					<img src={reactLogo} className='logo react' alt='React logo' />
				</a>
			</div>
			<h1>Vite + React</h1>
			<div className='card'>
				<button onClick={() => setCount((count) => count + 1)}>
					count is {count}
				</button>
				<p>
					Edit <code>src/App.tsx</code> and save to test HMR
				</p>
			</div>
			<p className='read-the-docs'>
				Click on the Vite and React logos to learn more
			</p>
		</>
	);
}

export default App;
