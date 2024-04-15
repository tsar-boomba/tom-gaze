#!/bin/sh

emcc -s MALLOC=emmalloc \
	-s MODULARIZE=0 \
	-s INCLUDE_FULL_LIBRARY=1 \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s ENVIRONMENT="web" $@
