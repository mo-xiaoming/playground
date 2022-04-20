import subprocess
import os
import sys
import json

out = subprocess.check_output(['g++', '-E', '-Wp,-v', '-xc++', '/dev/null'], stderr=subprocess.STDOUT)

started = False
includes = []
for line in out.splitlines():
    if line.startswith('#include <...> search starts here:'):
        started = True
    elif line.startswith('End of search list.'):
        break
    elif started:
        includes.append(line)
if includes:
    includes = " -isystem" + " -isystem".join(includes)
else:
    includes = ""

CC = sys.argv[1]

dcc = dict()
with open(CC) as f:
    dcc = json.load(f)

for i in dcc:
    i["command"] += includes

json.dump(dcc, sys.stdout, indent=2)
