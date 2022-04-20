#!/usr/bin/env bash

set -e
set -o pipefail

export CC=clang
export CXX=clang++

readonly TOP_OUT_DIR=build

echo '----------------------------------------------------'
echo '----------------- address sanitizer ----------------'
echo '----------------------------------------------------'

OUT_DIR=$TOP_OUT_DIR/address-clang
[[ -d $OUT_DIR ]] || mkdir -p $OUT_DIR

conan install . -s build_type=Debug -if $OUT_DIR -b missing
cmake -S. -B${OUT_DIR} -GNinja -DCMAKE_BUILD_TYPE=Debug -DSANITIZE_ADDRESS:BOOL=ON
cmake --build $OUT_DIR
ASAN_OPTIONS=strict_string_checks=1:detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1 $OUT_DIR/tests

echo '----------------------------------------------------'
echo '---------------- undefined sanitizer ---------------'
echo '----------------------------------------------------'

OUT_DIR=$TOP_OUT_DIR/undefined-clang
[[ -d $OUT_DIR ]] || mkdir -p $OUT_DIR

conan install . -s build_type=Debug -if $OUT_DIR -b missing
cmake -S. -B${OUT_DIR} -GNinja -DCMAKE_BUILD_TYPE=Debug -DSANITIZE_UNDEFINED:BOOL=ON
cmake --build $OUT_DIR
UBSAN_OPTIONS=print_stacktrace=1:suppressions=../../ubsan.supp $OUT_DIR/tests

if [[ ! -f compile_commands.json ]]; then
  [[ -f $OUT_DIR/compile_commands.json ]] && ln -sf $OUT_DIR/compile_commands.json .
fi

#run-clang-tidy $PWD -quiet
