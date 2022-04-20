#include <algorithm>
#include <limits>
#include <string>
#include <variant>

#include <gtest/gtest.h>

#include <mx.hpp>

TEST(RandomGenTest, Random) {
  using I = long long;
  auto rg = mx::RandomGen<std::numeric_limits<I>::min(),
                          std::numeric_limits<I>::max()>{};
  auto a = std::array<I, 10>{};

  std::generate(begin(a), end(a), [&rg] { return rg.next(); });
  std::sort(begin(a), end(a));
  EXPECT_EQ(std::adjacent_find(cbegin(a), cend(a)), cend(a));
}

TEST(RandomGenTest, EqRange) {
  auto rg = mx::RandomGen<3, 3>{};
  EXPECT_EQ(3, rg.next());
}

TEST(RandomGenTest, InRange) {
  auto rg = mx::RandomGen<-3, 3>{};
  for (int i = 0; i < 100; ++i) {
    const auto r = rg.next();
    EXPECT_TRUE(r >= -3 && r <= 3);
  }
}

TEST(LambdaOverloadedTest, Good) {
  struct S {};
  auto v = std::variant<int, std::string, S, long>{};
  auto f = mx::overloaded{[](auto) { return 3; }, [](S) { return 2; },
                          [](std::string) { return 1; }, [](int) { return 0; }};

  EXPECT_EQ(std::visit(f, v), 0);

  v = std::string{"hello"};
  EXPECT_EQ(std::visit(f, v), 1);

  v = S{};
  EXPECT_EQ(std::visit(f, v), 2);

  v = 3L;
  EXPECT_EQ(std::visit(f, v), 3);
}

TEST(ScopeExit, Success) {
  struct S {
    int i = 0;
  } s;
  {
    SCOPE_EXIT { s.i = 1; };
  }
  EXPECT_EQ(s.i, 1);
}

TEST(ScopeExit, Throw) {
  struct S {
    int i = 0;
  } s;
  try {
    SCOPE_EXIT { s.i = 1; };
    throw 3;
  } catch (...) {
  }
  EXPECT_EQ(s.i, 1);
}

TEST(ScopeFail, Throw) {
  struct S {
    int i = 0;
  } s;
  try {
    SCOPE_FAIL { s.i = 1; };
    throw 3;
  } catch (...) {
  }
  EXPECT_EQ(s.i, 1);
}

TEST(ScopeFail, Success) {
  struct S {
    int i = 0;
  } s;
  {
    SCOPE_FAIL { s.i = 1; };
  }
  EXPECT_EQ(s.i, 0);
}
