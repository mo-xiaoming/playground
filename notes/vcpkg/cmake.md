- spdlog
```
find_package(spdlog CONFIG REQUIRED)
target_link_libraries(main PRIVATE spdlog::spdlog spdlog::spdlog_header_only)
```

- fmt
```
find_package(fmt CONFIG REQUIRED)
target_link_libraries(main PRIVATE fmt::fmt fmt::fmt-header-only)
```

- benchmark
```
find_package(benchmark CONFIG REQUIRED)
target_link_libraries(main PRIVATE benchmark::benchmark benchmark::benchmark_main)
```

- gtest
```
enable_testing()

find_package(GTest CONFIG REQUIRED)
target_link_libraries(main PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)

add_test(AllTestsInMain main)
```

- catch2
```
find_package(Catch2 CONFIG REQUIRED)
target_link_libraries(main PRIVATE Catch2::Catch2)
```
