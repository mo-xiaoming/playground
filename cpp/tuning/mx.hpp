#include <random>
#include <vector>

#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace mx {

// repeatedly release/grow vectors
template <typename T, std::size_t InitCapacity> class CyclingVector {
public:
  CyclingVector();
  CyclingVector(const CyclingVector& rhs) : CyclingVector{} { v_ = rhs.v_; }
  auto operator=(const CyclingVector&) & -> CyclingVector& = delete;
  CyclingVector(CyclingVector&&) noexcept = delete;
  auto operator=(CyclingVector&&) & noexcept -> CyclingVector& = delete;
  ~CyclingVector();

private:
  static std::vector<std::vector<T>> recycling_;
  std::vector<T> v_;
};

template <typename T, std::size_t InitCapacity> CyclingVector<T, InitCapacity>::CyclingVector() {
  if (recycling_.empty()) {
    v_.reserve(InitCapacity);
  } else {
    v_ = std::move(recycling_.back());
    recycling_.pop_back();
  }
}

template <typename T, std::size_t InitCapacity> CyclingVector<T, InitCapacity>::~CyclingVector() {
  v_.clear();
  recycling_.push_back(std::move(v_));
}

template <typename T, std::size_t InitCapacity>
std::vector<std::vector<T>> CyclingVector<T, InitCapacity>::recycling_{};

// number randomizer
template <auto Min, decltype(Min) Max> class RandomGen {
public:
  [[nodiscard]] auto next() { return dis_(rng_); }

private:
  static std::random_device rd_;
  std::mt19937 rng_{rd_()};
  std::uniform_int_distribution<decltype(Min)> dis_{Min, Max};
};

template <auto Min, decltype(Min) Max> std::random_device RandomGen<Min, Max>::rd_{};

// number randomizer
template <typename T> class RandomGen1 {
public:
  RandomGen1(T Min, T Max) : dis_{Min, Max} {}
  [[nodiscard]] auto next() { return dis_(rng_); }

private:
  std::random_device rd_;
  std::mt19937 rng_{rd_()};
  std::uniform_int_distribution<T> dis_;
};

// uuid
class UuidGen {
public:
  using uuidType = boost::uuids::uuid;
  [[nodiscard]] static auto next() -> uuidType { return gen_(); }
  [[nodiscard]] static auto to_string(const uuidType& uuid) -> std::string { return boost::uuids::to_string(uuid); }

private:
  static boost::uuids::random_generator gen_;
};

// helper for std::variant, lambda overload
template <class... Ts> struct overloaded : Ts... {
  constexpr overloaded(Ts... ts) noexcept((std::is_nothrow_move_constructible_v<Ts> && ...)) : Ts(std::move(ts))... {}
  using Ts::operator()...;
};
template <class... Ts> overloaded(Ts...)->overloaded<Ts...>;

// for_each_argument([](auto i) { std::cout << i << '\n';}, 1, 3.7, "hello");
template <typename F, typename... Ts> void for_each_argument(F&& f, Ts&&... ts) {
  // (void) is here just incase `,` is overloaded
  ((void)std::forward<F>(f)(std::forward<Ts>(ts)), ...);
}

} // namespace mx
