/*

`c` for copy/create, `m` for move

// no alloc
bench_s_ref_short             2.53 ns 1c + 1c
bench_s_copy_short            2.55 ns 1c + 1c
bench_s_copy_move_short       7.04 ns 1c + 1m
bench_s_forward_short         7.01 ns 1c + 1m

// have alloc
bench_s_ref_long              37.7 ns 1c + 1c
bench_s_copy_long             38.1 ns 1c + 1c
bench_s_copy_move_long        19.0 ns 1c + 1m
bench_s_forward_long          18.9 ns 1c + 1m

// no alloc
bench_ref_short               2.50 ns 1c + 1c
bench_copy_short              2.49 ns 1c + 1c
bench_copy_move_short         7.01 ns 1c + 1m
bench_forward_short           1.80 ns 1c

// have alloc
bench_ref_long                37.7 ns 1c + 1c
bench_copy_long               37.8 ns 1c + 1c
bench_copy_move_long          19.0 ns 1c + 1m
bench_forward_long            19.1 ns 1c

// no alloc
bench_l_ref_short             8.37 ns 1c
bench_l_copy_short            16.3 ns 1c + 1c
bench_l_copy_move_short       10.8 ns 1c + 1m
bench_l_forward_short         8.43 ns 1c

// have alloc
bench_l_ref_long              25.6 ns 1c
bench_l_copy_long             50.4 ns 1c + 1c
bench_l_copy_move_long        25.1 ns 1c + 1m
bench_l_forward_long          25.1 ns 1c
*/

#include <string>

#include <benchmark/benchmark.h>

using namespace std::string_literals;

static void bench_s_ref_short(benchmark::State& state) {
  struct foo {
    foo(std::string const& s) : s{s} {}
    std::string s;
  };
  for (auto _ : state) {
    foo f{"short"s};
    benchmark::DoNotOptimize(f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_s_ref_short);

static void bench_s_copy_short(benchmark::State& state) {
  struct foo {
    foo(std::string s) : s{s} {}
    std::string s;
  };
  for (auto _ : state) {
    foo f{"short"s};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_s_copy_short);

static void bench_s_copy_move_short(benchmark::State& state) {
  struct foo {
    foo(std::string s) : s{std::move(s)} {}
    std::string s;
  };
  for (auto _ : state) {
    foo f{"short"s};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_s_copy_move_short);

struct fooT {
  template <typename T> fooT(T&& s) : s{std::forward<T>(s)} {}
  std::string s;
};
static void bench_s_forward_short(benchmark::State& state) {
  for (auto _ : state) {
    fooT f{"short"s};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_s_forward_short);

static void bench_s_ref_long(benchmark::State& state) {
  struct foo {
    foo(std::string const& s) : s{s} {}
    std::string s;
  };
  for (auto _ : state) {
    foo f{"looooooooooooooooooooooooooooooooooooooong"s};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_s_ref_long);

static void bench_s_copy_long(benchmark::State& state) {
  struct foo {
    foo(std::string s) : s{s} {}
    std::string s;
  };
  for (auto _ : state) {
    foo f{"looooooooooooooooooooooooooooooooooooooong"s};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_s_copy_long);

static void bench_s_copy_move_long(benchmark::State& state) {
  struct foo {
    foo(std::string s) : s{std::move(s)} {}
    std::string s;
  };
  for (auto _ : state) {
    foo f{"looooooooooooooooooooooooooooooooooooooong"s};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_s_copy_move_long);

static void bench_s_forward_long(benchmark::State& state) {
  for (auto _ : state) {
    fooT f{"looooooooooooooooooooooooooooooooooooooong"s};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_s_forward_long);

// const char *
static void bench_ref_short(benchmark::State& state) {
  struct foo {
    foo(std::string const& s) : s{s} {}
    std::string s;
  };
  for (auto _ : state) {
    foo f{"short"};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_ref_short);

static void bench_copy_short(benchmark::State& state) {
  struct foo {
    foo(std::string s) : s{s} {}
    std::string s;
  };
  for (auto _ : state) {
    foo f{"short"};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_copy_short);

static void bench_copy_move_short(benchmark::State& state) {
  struct foo {
    foo(std::string s) : s{std::move(s)} {}
    std::string s;
  };
  for (auto _ : state) {
    foo f{"short"};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_copy_move_short);

static void bench_forward_short(benchmark::State& state) {
  for (auto _ : state) {
    fooT f{"short"};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_forward_short);

static void bench_ref_long(benchmark::State& state) {
  struct foo {
    foo(std::string const& s) : s{s} {}
    std::string s;
  };
  for (auto _ : state) {
    foo f{"looooooooooooooooooooooooooooooooooooooong"};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_ref_long);

static void bench_copy_long(benchmark::State& state) {
  struct foo {
    foo(std::string s) : s{s} {}
    std::string s;
  };
  for (auto _ : state) {
    foo f{"looooooooooooooooooooooooooooooooooooooong"};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_copy_long);

static void bench_copy_move_long(benchmark::State& state) {
  struct foo {
    foo(std::string s) : s{std::move(s)} {}
    std::string s;
  };
  for (auto _ : state) {
    foo f{"looooooooooooooooooooooooooooooooooooooong"};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_copy_move_long);

static void bench_forward_long(benchmark::State& state) {
  for (auto _ : state) {
    fooT f{"looooooooooooooooooooooooooooooooooooooong"};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_forward_long);

// lvalue
static void bench_l_ref_short(benchmark::State& state) {
  struct foo {
    foo(std::string const& s) : s{s} {}
    std::string s;
  };
  std::string s{"short"};
  benchmark::DoNotOptimize(&s);
  for (auto _ : state) {
    foo f{s};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_l_ref_short);

static void bench_l_copy_short(benchmark::State& state) {
  struct foo {
    foo(std::string s) : s{s} {}
    std::string s;
  };
  std::string s{"short"};
  benchmark::DoNotOptimize(&s);
  for (auto _ : state) {
    foo f{s};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_l_copy_short);

static void bench_l_copy_move_short(benchmark::State& state) {
  struct foo {
    foo(std::string s) : s{std::move(s)} {}
    std::string s;
  };
  std::string s{"short"};
  benchmark::DoNotOptimize(&s);
  for (auto _ : state) {
    foo f{s};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_l_copy_move_short);

static void bench_l_forward_short(benchmark::State& state) {
  std::string s{"short"};
  benchmark::DoNotOptimize(&s);
  for (auto _ : state) {
    fooT f{s};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_l_forward_short);

static void bench_l_ref_long(benchmark::State& state) {
  struct foo {
    foo(std::string const& s) : s{s} {}
    std::string s;
  };
  std::string s{"looooooooooooooooooooooooooooooooooooooong"};
  benchmark::DoNotOptimize(&s);
  for (auto _ : state) {
    foo f{s};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_l_ref_long);

static void bench_l_copy_long(benchmark::State& state) {
  struct foo {
    foo(std::string s) : s{s} {}
    std::string s;
  };
  std::string s{"looooooooooooooooooooooooooooooooooooooong"};
  benchmark::DoNotOptimize(&s);
  for (auto _ : state) {
    foo f{s};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_l_copy_long);

static void bench_l_copy_move_long(benchmark::State& state) {
  struct foo {
    foo(std::string s) : s{std::move(s)} {}
    std::string s;
  };
  std::string s{"looooooooooooooooooooooooooooooooooooooong"};
  benchmark::DoNotOptimize(&s);
  for (auto _ : state) {
    foo f{s};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_l_copy_move_long);

static void bench_l_forward_long(benchmark::State& state) {
  std::string s{"looooooooooooooooooooooooooooooooooooooong"};
  benchmark::DoNotOptimize(&s);
  for (auto _ : state) {
    fooT f{s};
    benchmark::DoNotOptimize(&f);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_l_forward_long);

BENCHMARK_MAIN();
