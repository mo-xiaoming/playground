#include <gtest/gtest.h>

TEST(FooTest, TestMSan) {
  auto *a = new int;
  ASSERT_EQ(4, 4);
}
