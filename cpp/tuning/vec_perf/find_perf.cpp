#include <limits>
#include <random>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include <benchmark/benchmark.h>

#include <fmt/core.h>

struct U {
  std::unordered_set<int> u;
};

struct S {
  std::set<int> s;
};

struct V {
  explicit V(int64_t s) { v.reserve(s); }
  std::vector<int> v;
  void insert(int i);
};

void V::insert(int i) {
  for (auto it = v.begin(), end = v.end(); it != end; ++it) {
    if (i > *it) {
      v.insert(it, i);
      return;
    }
  }
  v.push_back(i);
}

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());

static void vector_1(benchmark::State& state) {
  auto v = V{state.range(0)};
  for (auto _ : state) {
    for (auto i = 0; i < state.range(0); ++i) {
      v.insert(dis(gen));
    }
    v.v.clear();
  }
}
BENCHMARK(vector_1)->Arg(1'000)->Arg(10'000)->Arg(100'000);

static void set_1(benchmark::State& state) {
  auto s = S{};
  for (auto _ : state) {
    for (auto i = 0; i < state.range(0); ++i) {
      s.s.insert(dis(gen));
    }
    s.s.clear();
  }
}
BENCHMARK(set_1)->Arg(1'000)->Arg(10'000)->Arg(100'000);

static void u_set_1(benchmark::State& state) {
  auto u = U{};
  for (auto _ : state) {
    for (auto i = 0; i < state.range(0); ++i) {
      u.u.insert(dis(gen));
    }
    u.u.clear();
  }
}
BENCHMARK(u_set_1)->Arg(1'000)->Arg(10'000)->Arg(100'000);

BENCHMARK_MAIN();
