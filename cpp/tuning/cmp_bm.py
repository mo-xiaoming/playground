#!/usr/bin/env python3

import json
import sys

funs = set()
args = set()
data = {} 

for bm in json.loads(sys.stdin.read())['benchmarks']:
    fn, *rest = bm['name'].split('/')
    funs.add(fn)
    args.add(tuple(int(i) for i in rest))
    data[bm['name']] = bm

funs = sorted(list(funs))
args = sorted(list(args))
#max_name = max(len(i) for i in funs)

for a in args:
    tag = "/".join(str(i) for i in [*a,])
    print(tag)
    base_time = data["/".join([funs[0], tag])]["cpu_time"]
    for f in funs:
        name = "/".join([f, tag])
        perf = 100.0
        if f != funs[0]:
            perf = data[name]["cpu_time"] / base_time * 100
        print("    {:20s}{:>10.2f}{}{:>10.2f}%".format(f, data[name]["cpu_time"], data[name]["time_unit"], perf)) 
