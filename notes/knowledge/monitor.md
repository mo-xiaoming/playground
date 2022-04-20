[top][top]
[vmstat][vmstat]
[free][free]
[mpstat][mpstat]
[pmap][pmap]

[top]: #top "top"
[vmstat]: #vmstat "vmstat"
[free]: #free "free"
[mpstat]: #mpstat "mpstat"
[pmap]: #pmap "pmap"


<a id="top"></a>top
===================

[ref](http://www.cyberciti.biz/tips/top-linux-monitoring-tools.html)

<dl>
<dt>t</dt><dd>Summary on/off</dd>
<dt>m</dt><dd>Memory on/off</dd>
<dt>A</dt><dd>Sorts by various top consumers</dd>
<dt>f</dt><dd>Configuration</dd>
<dt>r</dt><dd>renice</dd>
<dt>k</dt><dd>kill</dd>
<dt>z</dt><dd>color/mono</dd>
</dl>


<a id="vmstat"></a>vmstat
=========================

A better? choice - dstat

[ref](http://www.cyberciti.biz/tips/top-linux-monitoring-tools.html)

`vmstat` reports information about processes, memory, paging, block IO, traps, and cpu activity.

    procs -----------memory---------- ---swap-- -----io---- --system-- -----cpu------
     r  b   swpd   free   buff  cache      si   so    bi    bo   in   cs us sy id wa st
     0  0      0 2540988 522188 5130400    0    0     2    32    4    2  4  1 96  0  0
     1  0      0 2540988 522188 5130400    0    0     0   720 1199  665  1  0 99  0  0
     0  0      0 2540956 522188 5130400    0    0     0     0 1151 1569  4  1 95  0  0
     0  0      0 2540956 522188 5130500    0    0     0     6 1117  439  1  0 99  0  0
     0  0      0 2540940 522188 5130512    0    0     0   536 1189  932  1  0 98  0  0
     0  0      0 2538444 522188 5130588    0    0     0     0 1187 1417  4  1 96  0  0
     0  0      0 2490060 522188 5130640    0    0     0    18 1253 1123  5  1 94  0  0


<a id="free"></a>free
=========================

[ref](http://www.cyberciti.biz/tips/top-linux-monitoring-tools.html)

`free` displays the total amount of free and used physical and swap memory in the system, as well as the buffers used by the kernel.


    avg-cpu:  %user   %nice %system %iowait  %steal   %idle
               3.50    0.09    0.51    0.03    0.00   95.86
    Device:            tps   Blk_read/s   Blk_wrtn/s   Blk_read   Blk_wrtn
    sda              22.04        31.88       512.03   16193351  260102868
    sda1              0.00         0.00         0.00       2166        180
    sda2             22.04        31.87       512.03   16189010  260102688
    sda3              0.00         0.00         0.00       1615          0


<a id="mpstat"></a>mpstat
=========================

[ref](http://www.cyberciti.biz/tips/top-linux-monitoring-tools.html)

`mpstat` command displays activities for each available processor. `mpstat -P ALL` to display average CPU utilization per processor.

    08:02:55 AM  CPU    %usr   %nice    %sys %iowait    %irq   %soft  %steal  %guest   %idle
    08:02:55 AM  all   14.95    0.04   76.03    1.99    0.00    0.14    0.00    0.00    6.84
    08:02:55 AM    0   14.93    0.03   75.97    1.94    0.00    0.16    0.00    0.00    6.96
    08:02:55 AM    1   14.97    0.06   76.09    2.04    0.00    0.11    0.00    0.00    6.73


<a id="pmap"></a>pmap
=========================

[ref](http://www.cyberciti.biz/tips/top-linux-monitoring-tools.html)

`pmap` report memory map of a process. Use this command to find out causes of memory bottlenecks.

    47394:   /usr/bin/php-cgi
    Address           Kbytes Mode  Offset           Device    Mapping
    0000000000400000    2584 r-x-- 0000000000000000 008:00002 php-cgi
    0000000000886000     140 rw--- 0000000000286000 008:00002 php-cgi
    00000000008a9000      52 rw--- 00000000008a9000 000:00000   [ anon ]
    0000000000aa8000      76 rw--- 00000000002a8000 008:00002 php-cgi
    000000000f678000    1980 rw--- 000000000f678000 000:00000   [ anon ]
    000000314a600000     112 r-x-- 0000000000000000 008:00002 ld-2.5.so
    000000314a81b000       4 r---- 000000000001b000 008:00002 ld-2.5.so
    000000314a81c000       4 rw--- 000000000001c000 008:00002 ld-2.5.so
    000000314aa00000    1328 r-x-- 0000000000000000 008:00002 libc-2.5.so
    000000314ab4c000    2048 ----- 000000000014c000 008:00002 libc-2.5.so

<a id="iptraf"></a>iptraf
=========================

[ref](http://www.cyberciti.biz/tips/top-linux-monitoring-tools.html)

`iptraf` command is interactive colorful IP LAN monitor.
