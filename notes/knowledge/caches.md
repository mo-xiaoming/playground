Copied from [CppCon 2014](http://cppcon.org/2014-videos-online/): [Chandler Carruth "Efficiency with Algorithms, Performance with Data Structures"](https://www.youtube.com/watch?v=fHNmRkzxHWs) near 37:50

type                                |         time    | time    | relative
------------------------------------|-----------------|---------|-------------------
Once cycle on a 3 GHz rocessor      |           1 ns  |         |
L1 cache reference                  |         0.5 ns  |         |
Branch mispredict                   |           5 ns  |         |
L2 cache reference                  |           7 ns  |         | 14x L1 cache
Mutex lock/unlock                   |          25 ns  |         |
Main memory refrence                |         100 ns  |         | 20x L2, 200x L1
Compress 1k bytes with Snappy       |       3,000 ns  |         |
Send 1k bytes over 1 Gbps network   |      10,000 ns  | 0.01 ms |
Read 4k randomly from SSD*          |     150,000 ns  | 0.15 ms |
Read 1 MB sequentially from memory  |     250,000 ns  | 0.25 ms |
Round trip within same datacenter   |     500,000 ns  |  0.5 ms |
Read 1 MB sequentially from SSD*    |   1,000,000 ns  |    1 ms |  4x memory
Disk seek                           |  10,000,000 ns  |   10 ms | 20x datacenter RT
Read 1 MB sequentially from disk    |  20,000,000 ns  |   20 ms | 80x memory, 20x SSD
Send packet CA->Netherlands->CA     | 150,000,000 ns  |  150 ms |

Copied from [code - -dive conference 2014 - Scott Meyers - Cpu Caches and Why You Care](https://youtu.be/WDIkqP4JbkE) neer 16:59

For Core i7-9xx

type | latency (cycles)
-----|-----------------
L1   | 4
L2   | 11
L3   | 39
Main Memory|107
