/*

`bench_data`:
       717,040,256      L1-dcache-loads           # 1072.944 M/sec                    (71.74%)
         4,439,286      L1-dcache-load-misses     #    0.62% of all L1-dcache hits    (70.48%)

`bench_objo`:
       245,988,557      L1-dcache-loads           #  341.951 M/sec                    (71.66%)
       268,262,751      L1-dcache-load-misses     #  109.05% of all L1-dcache hits    (70.97%)

4
    bench_data                1.90ns    100.00%
    bench_objo                6.02ns    317.05%
5
    bench_data                5.17ns    100.00%
    bench_objo               11.91ns    230.45%
6
    bench_data                9.53ns    100.00%
    bench_objo               24.99ns    262.08%
7
    bench_data               19.41ns    100.00%
    bench_objo               46.92ns    241.70%
8
    bench_data               34.11ns    100.00%
    bench_objo               85.85ns    251.71%
9
    bench_data               57.34ns    100.00%
    bench_objo              181.39ns    316.32%
10
    bench_data              103.42ns    100.00%
    bench_objo             1194.71ns   1155.22%
11
    bench_data              200.35ns    100.00%
    bench_objo             2473.34ns   1234.52%
12
    bench_data              388.90ns    100.00%
    bench_objo             5420.23ns   1393.74%
*/

#include <algorithm>

#include <benchmark/benchmark.h>

#include "mx.hpp"

class S {
  int v00 = 7;
  int v01 = 7;
  int v02 = 7;
  int v03 = 7;
  int v04 = 7;
  int v05 = 7;
  int v06 = 7;
  int v07 = 7;
  int v08 = 7;
  int v09 = 7;
  int v10 = 7;
  int v11 = 7;
  int v12 = 7;
  int v13 = 7;
  int v14 = 7;
  int v15 = 7;

public:
  void update_v00() { v00 += 3; }
};

class V {
public:
  explicit V(int64_t s) {
    v00s.resize(s, 7);
    v01s.resize(s, 7);
    v02s.resize(s, 7);
    v03s.resize(s, 7);
    v04s.resize(s, 7);
    v05s.resize(s, 7);
    v06s.resize(s, 7);
    v07s.resize(s, 7);
    v08s.resize(s, 7);
    v09s.resize(s, 7);
    v11s.resize(s, 7);
    v12s.resize(s, 7);
    v13s.resize(s, 7);
    v14s.resize(s, 7);
    v15s.resize(s, 7);
  }
  void update_v00s() {
    std::transform(cbegin(v00s), cend(v00s), begin(v00s), [](auto i) { return i + 3; });
  }

private:
  std::vector<int> v00s{};
  std::vector<int> v01s{};
  std::vector<int> v02s{};
  std::vector<int> v03s{};
  std::vector<int> v04s{};
  std::vector<int> v05s{};
  std::vector<int> v06s{};
  std::vector<int> v07s{};
  std::vector<int> v08s{};
  std::vector<int> v09s{};
  std::vector<int> v10s{};
  std::vector<int> v11s{};
  std::vector<int> v12s{};
  std::vector<int> v13s{};
  std::vector<int> v14s{};
  std::vector<int> v15s{};
};

static void bench_objo(benchmark::State& state) {
  auto s = std::vector<S>{};
  s.resize(1 << state.range(0));
  for (auto _ : state) {
    std::for_each(begin(s), end(s), [](auto& i) { i.update_v00(); });
  }
}
BENCHMARK(bench_objo)->DenseRange(4, 12);

static void bench_data(benchmark::State& state) {
  auto v = V{1 << state.range(0)};
  for (auto _ : state) {
    v.update_v00s();
  }
}
BENCHMARK(bench_data)->DenseRange(4, 12);

BENCHMARK_MAIN();
