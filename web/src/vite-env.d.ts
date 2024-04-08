/* eslint-disable no-var */
/// <reference types="vite/client" />

interface Window {
	Module: EmscriptenModule;
	_malloc: (bytes: number) => number;
}

declare var Module: EmscriptenModule & {
	main: () => void;
	_infer: (buffer: number, len: number) => void;
};
declare var _malloc: (bytes: number) => number;
