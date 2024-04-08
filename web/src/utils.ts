export const moveToHeap = (buffer: Uint8Array): number => {
	const ptr = Module._malloc(buffer.byteLength);
	const dataOnHeap = new Uint8Array(Module.HEAPU8.buffer, ptr, buffer.byteLength);
	dataOnHeap.set(buffer);
	return ptr;
}
