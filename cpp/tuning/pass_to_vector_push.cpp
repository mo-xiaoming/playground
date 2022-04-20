/*

bench_short_ref          47.7 ns         47.2 ns     15041913
bench_short_list         39.7 ns         39.5 ns     18030594 *
bench_short_eb           40.3 ns         40.1 ns     17577095
bench_short_ref_eb       39.6 ns         39.3 ns     17445120

bench_long_ref           96.8 ns         96.0 ns      7061259
bench_long_list          54.7 ns         54.4 ns     11402972 *
bench_long_eb            57.8 ns         57.5 ns     11365074
bench_long_ref_eb        58.2 ns         57.9 ns     11653916

*/

#include <string>
#include <vector>

#include <benchmark/benchmark.h>

struct foo {
  template <typename T> foo(T&& t) : s{std::forward<T>(t)} {}
  std::string s;
};

static void bench_short_ref(benchmark::State& state) {
  struct bar {
    std::vector<foo> foos;
    void add(foo const& f) { foos.push_back(f); }
  };
  for (auto _ : state) {
    bar b{};
    benchmark::DoNotOptimize(&b.foos);
    b.add(foo{"hello"});
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_short_ref);

static void bench_short_list(benchmark::State& state) {
  struct bar {
    std::vector<foo> foos;
    void add(const char* s) { foos.push_back({s}); }
  };
  for (auto _ : state) {
    bar b{};
    benchmark::DoNotOptimize(&b.foos);
    b.add("hello");
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_short_list);

static void bench_short_eb(benchmark::State& state) {
  struct bar {
    std::vector<foo> foos;
    void add(const char* s) { foos.emplace_back(s); }
  };
  for (auto _ : state) {
    bar b{};
    benchmark::DoNotOptimize(&b.foos);
    b.add("hello");
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_short_eb);

struct barT {
  std::vector<foo> foos;
  template <typename... T> void add(T&&... t) { foos.emplace_back(std::forward<T>(t)...); }
};
static void bench_short_ref_eb(benchmark::State& state) {
  for (auto _ : state) {
    barT b{};
    benchmark::DoNotOptimize(&b.foos);
    b.add("hello");
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_short_ref_eb);

static void bench_long_ref(benchmark::State& state) {
  struct bar {
    std::vector<foo> foos;
    void add(foo const& f) { foos.push_back(f); }
  };
  for (auto _ : state) {
    bar b{};
    benchmark::DoNotOptimize(&b.foos);
    b.add(foo{"loooooooooooooooooooooooooooooooooooooong"});
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_long_ref);

static void bench_long_list(benchmark::State& state) {
  struct bar {
    std::vector<foo> foos;
    void add(const char* s) { foos.push_back({s}); }
  };
  for (auto _ : state) {
    bar b{};
    benchmark::DoNotOptimize(&b.foos);
    b.add("loooooooooooooooooooooooooooooooooooooong");
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_long_list);

static void bench_long_eb(benchmark::State& state) {
  struct bar {
    std::vector<foo> foos;
    void add(const char* s) { foos.emplace_back(s); }
  };
  for (auto _ : state) {
    bar b{};
    benchmark::DoNotOptimize(&b.foos);
    b.add("loooooooooooooooooooooooooooooooooooooong");
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_long_eb);

static void bench_long_ref_eb(benchmark::State& state) {
  for (auto _ : state) {
    barT b{};
    benchmark::DoNotOptimize(&b.foos);
    b.add("loooooooooooooooooooooooooooooooooooooong");
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_long_ref_eb);

BENCHMARK_MAIN();
