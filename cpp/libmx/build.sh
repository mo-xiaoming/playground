#!/bin/sh

rm -rf build
cmake -H. -Bbuild -DCMAKE_PREFIX_PATH=/home/mx/.local/third_parties -DCMAKE_INSTALL_PREFIX=/home/mx/.local/third_parties &&
	cmake --build build -- VERBOSE=1 &&
	cmake --build build --target test &&
	cmake --build build --target memcheck
