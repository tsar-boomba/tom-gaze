#!/bin/sh

set -e
export RUSTFLAGS="-C linker=$(pwd)/emcc-link.sh"
cargo build -p web-infer --target wasm32-unknown-emscripten --release

TARGET=target/wasm32-unknown-emscripten/release
cp $TARGET/web-infer.js web/public/web-infer.js
cp $TARGET/web_infer.wasm web/public/web_infer.wasm
