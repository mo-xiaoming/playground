#!/usr/bin/env bash

mkdir -p build/cc_json

for i in $(find -name \*.cpp -a ! -path './build/*'); do
	clang++ -MJ build/cc_json/${i%cpp}o.json -Wall -std=c++11 -o ${i%cpp}o -c $i
done

sed -e '1s/^/[\n/' -e '$s/,$/\n]/' $(find build/cc_json -name \*.o.json) > compile_commands.json

rm -rf build/cc_json
