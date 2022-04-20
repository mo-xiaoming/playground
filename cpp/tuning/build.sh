#!/bin/sh

# cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr
# -DCMAKE_INSTALL_PREFIX=
# cmake --build build
# make DESTDIR=/home/john install
# cmake --build build --target install --config Release
# cmake --install <dir> --prefix "/usr"

cmake -B build -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_INSTALL_PREFIX:PATH=/home/mx/.local/third_parties &&
	cmake --build build
