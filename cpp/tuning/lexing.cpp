// no performance difference,
// maybe andrei Alexandrescu is wrong about 5x performance improvement?
// Declarative Control Flow
#include <algorithm>

#include <benchmark/benchmark.h>

#include "mx.hpp"

volatile int i = 0;

void foo1() { i = 2; }
void foo2() { i = 3; }
void foo3() { i = 4; }
void foo4() { i = 5; }
void foo5() { i = 6; }
void foo6() { i = 7; }
void foo7() { i = 8; }
void foo8() { i = 9; }
void foo9() { i = 0; }
void foo0() { i = 10; }
void foo10() { i = 11; }

void process_switch(char* const buf, int len) {
  auto const save = buf[len - 1];
  buf[len - 1] = -1;
  for (auto p = buf;;) {
    switch (auto c = *p++; c) {
    case 0:
      foo1();
      break;
    case 1:
      foo2();
      break;
    case 2:
      foo3();
      break;
    case 3:
      foo4();
      break;
    case 4:
      foo5();
      break;
    case 5:
      foo6();
      break;
    case 6:
      foo7();
      break;
    case 7:
      foo8();
      break;
    case 8:
      foo9();
      break;
    case 9:
      foo0();
      break;
    case -1:
      foo10();
      buf[len - 1] = save;
      return;
    }
  }
}

void process_if(char* const buf, int len) {
  for (int i = 0; i < len; ++i) {
    auto c = buf[i];
    if (c == 0) {
      foo1();
    } else if (c == 1) {
      foo2();
    } else if (c == 2) {
      foo3();
    } else if (c == 3) {
      foo4();
    } else if (c == 4) {
      foo5();
    } else if (c == 5) {
      foo6();
    } else if (c == 6) {
      foo7();
    } else if (c == 7) {
      foo8();
    } else if (c == 8) {
      foo9();
    } else if (c == 9) {
      foo0();
    } else {
      foo10();
    }
  }
}

auto rg = mx::RandomGen<static_cast<char>(0), static_cast<char>(9)>{};

#define C(f)                                                                                                           \
  do {                                                                                                                 \
    for (auto _ : state) {                                                                                             \
      state.PauseTiming();                                                                                             \
      auto v = std::vector<char>{};                                                                                    \
      auto const size = state.range(0);                                                                                \
      v.reserve(size);                                                                                                 \
      std::generate_n(back_inserter(v), size, [] { return rg.next(); });                                               \
      state.ResumeTiming();                                                                                            \
      f(v.data(), size);                                                                                               \
    }                                                                                                                  \
  } while (0)

static void bench_if(benchmark::State& state) { C(process_if); }
BENCHMARK(bench_if)->Range(10, 1000);

static void bench_switch(benchmark::State& state) { C(process_switch); }
BENCHMARK(bench_switch)->Range(10, 1000);

BENCHMARK_MAIN();
