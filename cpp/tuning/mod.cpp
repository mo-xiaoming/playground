#include <algorithm>

#include <benchmark/benchmark.h>

#include "mx.hpp"

#define unlikely(x) __builtin_expect(!!(x), 0)

static void bench_mod_raw(benchmark::State& state) {
  auto a = std::vector<unsigned char>{};
  a.reserve(1024);
  std::generate_n(back_inserter(a), 1024, [rg = mx::RandomGen<0, 255>{}]() mutable { return rg.next(); });
  auto const top = state.range(0);
  for (auto _ : state) {
    for (auto i : a) {
      unsigned char const r = i % top;
      benchmark::DoNotOptimize(&r);
    }
  }
}
BENCHMARK(bench_mod_raw)->Arg(1)->Arg(64)->Arg(128)->Arg(192)->Arg(256);

#define MODRAW(N)                                                                                                      \
  do {                                                                                                                 \
    unsigned char const r = a[i + (N)] % top;                                                                            \
    benchmark::DoNotOptimize(&r);                                                                                      \
  } while (0)

static void bench_mod_raw16(benchmark::State& state) {
  auto a = std::vector<unsigned char>{};
  a.reserve(1024);
  std::generate_n(back_inserter(a), 1024, [rg = mx::RandomGen<0, 255>{}]() mutable { return rg.next(); });
  auto const top = state.range(0);
  for (auto _ : state) {
    for (auto i = 0U; i < a.size(); i += 16) {
      MODRAW(0);
      MODRAW(1);
      MODRAW(2);
      MODRAW(3);
      MODRAW(4);
      MODRAW(5);
      MODRAW(6);
      MODRAW(7);
      MODRAW(8);
      MODRAW(9);
      MODRAW(10);
      MODRAW(11);
      MODRAW(12);
      MODRAW(13);
      MODRAW(14);
      MODRAW(15);
    }
  }
}
BENCHMARK(bench_mod_raw16)->Arg(1)->Arg(64)->Arg(128)->Arg(192)->Arg(256);

static auto mod(unsigned char i, int16_t top) -> unsigned char { return i % top; }

static void bench_mod(benchmark::State& state) {
  auto a = std::vector<unsigned char>{};
  a.reserve(1024);
  std::generate_n(back_inserter(a), 1024, [rg = mx::RandomGen<0, 255>{}]() mutable { return rg.next(); });
  auto const top = state.range(0);
  for (auto _ : state) {
    for (auto i : a) {
      unsigned char const r = mod(i, top);
      benchmark::DoNotOptimize(&r);
    }
  }
}
BENCHMARK(bench_mod)->Arg(1)->Arg(64)->Arg(128)->Arg(192)->Arg(256);

#define MOD(N)                                                                                                         \
  do {                                                                                                                 \
    unsigned char const r = mod(a[i + (N)], top);                                                                        \
    benchmark::DoNotOptimize(&r);                                                                                      \
  } while (0)

static void bench_mod16(benchmark::State& state) {
  auto a = std::vector<unsigned char>{};
  a.reserve(1024);
  std::generate_n(back_inserter(a), 1024, [rg = mx::RandomGen<0, 255>{}]() mutable { return rg.next(); });
  auto const top = state.range(0);
  for (auto _ : state) {
    for (auto i = 0U; i < a.size(); i += 16) {
      MOD(0);
      MOD(1);
      MOD(2);
      MOD(3);
      MOD(4);
      MOD(5);
      MOD(6);
      MOD(7);
      MOD(8);
      MOD(9);
      MOD(10);
      MOD(11);
      MOD(12);
      MOD(13);
      MOD(14);
      MOD(15);
    }
  }
}
BENCHMARK(bench_mod16)->Arg(1)->Arg(64)->Arg(128)->Arg(192)->Arg(256);

static auto fastmod(unsigned char i, int16_t top) -> unsigned char {
  if (i >= top) {
    return i % top;
  }
  return i;
}

static void bench_fastmod(benchmark::State& state) {
  auto a = std::vector<unsigned char>{};
  a.reserve(1024);
  std::generate_n(back_inserter(a), 1024, [rg = mx::RandomGen<0, 255>{}]() mutable { return rg.next(); });
  auto const top = state.range(0);
  for (auto _ : state) {
    for (auto i : a) {
      unsigned char const r = fastmod(i, top);
      benchmark::DoNotOptimize(&r);
    }
  }
}
BENCHMARK(bench_fastmod)->Arg(1)->Arg(64)->Arg(128)->Arg(192)->Arg(256);

#define FASTMOD(N, F)                                                                                                  \
  do {                                                                                                                 \
    unsigned char const r = F(a[i + (N)], top);                                                                          \
    benchmark::DoNotOptimize(&r);                                                                                      \
  } while (0)

static void bench_fastmod16(benchmark::State& state) {
  auto a = std::vector<unsigned char>{};
  a.reserve(1024);
  std::generate_n(back_inserter(a), 1024, [rg = mx::RandomGen<0, 255>{}]() mutable { return rg.next(); });
  auto const top = state.range(0);
  for (auto _ : state) {
    for (auto i = 0U; i < a.size(); i += 16) {
      FASTMOD(0, fastmod);
      FASTMOD(1, fastmod);
      FASTMOD(2, fastmod);
      FASTMOD(3, fastmod);
      FASTMOD(4, fastmod);
      FASTMOD(5, fastmod);
      FASTMOD(6, fastmod);
      FASTMOD(7, fastmod);
      FASTMOD(8, fastmod);
      FASTMOD(9, fastmod);
      FASTMOD(10, fastmod);
      FASTMOD(11, fastmod);
      FASTMOD(12, fastmod);
      FASTMOD(13, fastmod);
      FASTMOD(14, fastmod);
      FASTMOD(15, fastmod);
    }
  }
}
BENCHMARK(bench_fastmod16)->Arg(1)->Arg(64)->Arg(128)->Arg(192)->Arg(256);

static auto fastmod_u(unsigned char i, int16_t top) -> unsigned char {
  if (unlikely(i >= top)) {
    return i % top;
  }
  return i;
}
static void bench_fastmod16_u(benchmark::State& state) {
  auto a = std::vector<unsigned char>{};
  a.reserve(1024);
  std::generate_n(back_inserter(a), 1024, [rg = mx::RandomGen<0, 255>{}]() mutable { return rg.next(); });
  auto const top = state.range(0);
  for (auto _ : state) {
    for (auto i = 0U; i < a.size(); i += 16) {
      FASTMOD(0, fastmod_u);
      FASTMOD(1, fastmod_u);
      FASTMOD(2, fastmod_u);
      FASTMOD(3, fastmod_u);
      FASTMOD(4, fastmod_u);
      FASTMOD(5, fastmod_u);
      FASTMOD(6, fastmod_u);
      FASTMOD(7, fastmod_u);
      FASTMOD(8, fastmod_u);
      FASTMOD(9, fastmod_u);
      FASTMOD(10, fastmod_u);
      FASTMOD(11, fastmod_u);
      FASTMOD(12, fastmod_u);
      FASTMOD(13, fastmod_u);
      FASTMOD(14, fastmod_u);
      FASTMOD(15, fastmod_u);
    }
  }
}
BENCHMARK(bench_fastmod16_u)->Arg(1)->Arg(64)->Arg(128)->Arg(192)->Arg(256);

BENCHMARK_MAIN();
