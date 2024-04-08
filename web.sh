#!/bin/sh

set -e
RUSTFLAGS="-C linker=$(pwd)/emcc-link.sh"
cargo build -p web-infer --target wasm32-unknown-emscripten --release

rm -rf web-target
mkdir web-target

TARGET=target/wasm32-unknown-emscripten/release
cp $TARGET/web-infer.js web/public/web-infer.js
cp $TARGET/web_infer.wasm web/public/web_infer.wasm
