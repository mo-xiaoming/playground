#include <atomic>
#include <iostream>
#include <mutex>
#include <new>
#include <thread>

#include <benchmark/benchmark.h>

#include "mx.hpp"

namespace {
void work(std::atomic<int>& a) {
  for (auto i = 0; i < 10'000'000; ++i) {
    ++a;
  }
}

void work(int& a) {
  for (auto i = 0; i < 10'000'000; ++i) {
    ++a;
  }
}
} // namespace

static void bench_false_sharing(benchmark::State& state) {
  for (auto _ : state) {
    int a{};
    int b{};
    int c{};
    int d{};

    std::thread t1{[&a] { work(a); }};
    std::thread t2{[&b] { work(b); }};
    std::thread t3{[&c] { work(c); }};
    std::thread t4{[&d] { work(d); }};

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    auto sum = a + b + c + d;
    benchmark::DoNotOptimize(&sum);
  }
}
BENCHMARK(bench_false_sharing)->DisplayAggregatesOnly(true);

static void bench_sharing(benchmark::State& state) {
  for (auto _ : state) {
    std::atomic<int> sum = 0;
    std::thread t1{[&sum] { work(sum); }};
    std::thread t2{[&sum] { work(sum); }};
    std::thread t3{[&sum] { work(sum); }};
    std::thread t4{[&sum] { work(sum); }};

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    benchmark::DoNotOptimize(&sum);
  }
}
BENCHMARK(bench_sharing)->DisplayAggregatesOnly(true);

static void bench(benchmark::State& state) {
  for (auto _ : state) {
    struct alignas(64) A {
      int a{};
    } a, b, c, d;

    std::thread t1{[&a] { work(a.a); }};
    std::thread t2{[&b] { work(b.a); }};
    std::thread t3{[&c] { work(c.a); }};
    std::thread t4{[&d] { work(d.a); }};

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    int sum = a.a + b.a + c.a + d.a;
    benchmark::DoNotOptimize(&sum);
  }
}
BENCHMARK(bench)->DisplayAggregatesOnly(true);

BENCHMARK_MAIN();
