# get cpu architecture family

`gcc -march=native -Q --help=target |grep march`

`cat /sys/devices/cpu/caps/pmu_name`
