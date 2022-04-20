```
==11603==ERROR: AddressSanitizer failed to allocate 0x200001000 (8589938688) bytes at address ffffff000 (errno: 12)
==11603==ReserveShadowMemoryRange failed while trying to map 0x200001000 bytes. Perhaps you're using ulimit -v
Aborted
```

`echo 0 > /proc/sys/vm/overcommit_memory` solves it, which value should have been `2`

