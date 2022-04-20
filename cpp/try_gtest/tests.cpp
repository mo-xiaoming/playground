#include <deque>
#include <limits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

TEST(VectorTest, Compare) { // NOLINT
    auto const x = std::vector{1, 2, 3};
    auto const y = std::vector{1, 2, 3};

    ASSERT_EQ(x.size(), y.size());

    for (auto i = 0U; i < x.size(); ++i) {
        EXPECT_EQ(x[i], y[i]) << "Vectors x and y differ at index " << i;
    }
}

TEST(ValuesTest, Compare) { // NOLINT
    constexpr auto expected = 3;
    EXPECT_EQ(3, expected); // ==
    EXPECT_NE(0, expected); // !=
    EXPECT_LT(0, expected); // <
    EXPECT_LE(3, expected); // <=
    EXPECT_GT(4, expected); // >
    EXPECT_GE(3, expected); // >=
}

TEST(RawStringTest, Compare) { // NOLINT
    constexpr char const* const expected = "hello";
    EXPECT_STRCASEEQ("Hello", expected);
    EXPECT_STRCASENE("", expected);
    EXPECT_STREQ("hello", expected);
    EXPECT_STRNE("Hello", expected);
}

template <typename E> class Queue {
public:
    void equeue(E e) {
        elems_.push_back(new E(e)); // NOLINT
    }
    E* dequeue() {
        if (elems_.empty()) {
            return nullptr;
        }
        auto* e = elems_.front();
        elems_.pop_front();
        return e;
    }
    [[nodiscard]] int64_t size() const {
        return static_cast<int64_t>(elems_.size());
    }

    void purge() {
        for (auto* e : elems_) {
            delete e; // NOLINT
        }
    }

private:
    std::deque<E*> elems_;
};

class QueueTest : public ::testing::Test { // NOLINT
protected:
    void SetUp() override {
        q1_.equeue(1);
        q2_.equeue(2);
        q2_.equeue(3);
    }
    void TearDown() override;

    Queue<int> q0_; // NOLINT
    Queue<int> q1_; // NOLINT
    Queue<int> q2_; // NOLINT
};

void QueueTest::TearDown() {}

TEST_F(QueueTest, IsEmptyInitially) { // NOLINT
    EXPECT_EQ(q0_.size(), 0);
    q0_.purge();
    q1_.purge();
    q2_.purge();
}

TEST_F(QueueTest, DequeueWorks) { // NOLINT
    auto* n = q0_.dequeue();
    EXPECT_EQ(n, nullptr);

    n = q1_.dequeue();
    ASSERT_NE(n, nullptr);
    EXPECT_EQ(*n, 1);
    EXPECT_EQ(q1_.size(), 0);
    delete n; // NOLINT

    n = q2_.dequeue();
    ASSERT_NE(n, nullptr);
    EXPECT_EQ(*n, 2);
    delete n; // NOLINT
    EXPECT_EQ(q2_.size(), 1);
    n = q2_.dequeue();
    EXPECT_EQ(*n, 3);
    delete n; // NOLINT
    EXPECT_EQ(q2_.size(), 0);
}

struct Some_container {
    [[nodiscard]] constexpr std::uint64_t size() const {
        return std::numeric_limits<std::uint64_t>::max();
    }
};

struct ContainerTwoLarge : std::exception {
    explicit ContainerTwoLarge(std::string_view msg) : msg_(msg) {}
    [[nodiscard]] char const* what() const noexcept override;

private:
    std::string msg_;
};

char const* ContainerTwoLarge::what() const noexcept { return msg_.data(); }

static const uint64_t max_size =
    static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
struct X {
    [[nodiscard]] constexpr std::int64_t size() const {
        if (container_.size() > max_size) {
            throw ContainerTwoLarge("X underlying structure is too large");
        }
        return static_cast<std::int64_t>(container_.size());
    }

private:
    Some_container container_;
};

TEST(IntegerTest, Overflow) { // NOLINT
    try {
        auto a = X().size();
        ASSERT_EQ(a, 0);
    } catch (ContainerTwoLarge const&) {
    }
}

struct Arg {
    Arg() = default;
    Arg(Arg const&) = default;
    Arg& operator=(Arg const&) noexcept = default;
    Arg(Arg&&) = default;
    Arg& operator=(Arg&&) noexcept = default;
    virtual ~Arg();

    [[nodiscard]] virtual constexpr int size() const noexcept = 0;
    virtual bool empty() = 0;
    virtual void attr(int) = 0;
};

Arg::~Arg() = default;

struct Mock_arg : Arg { // NOLINT
    ~Mock_arg() override;
    MOCK_METHOD(int, size, (), (override, const, noexcept)); // NOLINT
    MOCK_METHOD(bool, empty, (), (override));                // NOLINT
    MOCK_METHOD(void, attr, (int), (override));              // NOLINT
};

Mock_arg::~Mock_arg() = default;

static bool call_arg(Arg* arg) {
    auto s = arg->size();
    arg->attr(s);
    arg->attr(s + 1);
    return arg->empty();
}

using ::testing::AtLeast;
using ::testing::Return;

TEST(MockTest, SmallMock) { // NOLINT
    auto arg = Mock_arg();
    EXPECT_CALL(arg, empty()).Times(AtLeast(1)).WillRepeatedly(Return(false));
    EXPECT_CALL(arg, size).Times(1).WillOnce(Return(0));
    EXPECT_CALL(arg, attr(::testing::_)).Times(2);

    call_arg(&arg);
}

struct Mock_tmpl_arg : Arg { // NOLINT
    ~Mock_tmpl_arg() override;
    MOCK_METHOD(int, size, (), (const, noexcept)); // NOLINT
    MOCK_METHOD(bool, empty, (), ());                // NOLINT
    MOCK_METHOD(void, attr, (int), ());              // NOLINT
};

Mock_tmpl_arg::~Mock_tmpl_arg() = default;

template <typename Arg_type>
static bool call_tmpl_arg(Arg_type * arg) {
    auto s = arg->size();
    arg->attr(s);
    arg->attr(s + 1);
    return arg->empty();
}

TEST(MockTest, SmallTemplateMock) { // NOLINT
    auto arg = Mock_tmpl_arg();
    EXPECT_CALL(arg, empty()).Times(AtLeast(1)).WillRepeatedly(Return(false));
    EXPECT_CALL(arg, size).Times(1).WillOnce(Return(0));
    EXPECT_CALL(arg, attr(::testing::_)).Times(2);

    call_tmpl_arg(&arg);
}
