#!/bin/sh

emcc -s MALLOC=emmalloc \
-s "EXPORTED_FUNCTIONS=['_main', '_infer', 'malloc', '_malloc']" \
-s ENVIRONMENT="web" \
-s EXPORT_ES6=1 \
-s INCLUDE_FULL_LIBRARY=1 \
-s USE_ES6_IMPORT_META=1 $@
