// https://godbolt.org/z/KaG9f8EWj

#include <memory>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

uint64_t checkForErrorA();
uint64_t checkForErrorB();
void handleErrorA();
void handleErrorB();

void sendOrderToExchange();
__attribute__((noinline)) void handleError(uint64_t);

void bad0() {
  if (checkForErrorA())
    handleErrorA();
  else if (checkForErrorB())
    handleErrorB();
  else
    sendOrderToExchange();
}

void good0() {
  uint64_t errorFlags = 0;

  errorFlags |= checkForErrorA();
  errorFlags |= checkForErrorB();

  if (likely(!errorFlags))
    sendOrderToExchange();
  else
    handleError(errorFlags);
}
///////////////////////////////////////////////////

enum class Side { Buy, Sell };

template <Side T> struct Strategy {
  void RunStrategy(float fairValue, float credit);
  float CalcPrice(float fairValue, float credit);
  void SendOrder(float orderPrice);
};

template <Side T> void Strategy<T>::RunStrategy(float fairValue, float credit) {
  const float orderPrice = CalcPrice(fairValue, credit);
  SendOrder(orderPrice);
}

template <>
float Strategy<Side::Buy>::CalcPrice(float fairValue, float credit) {
  return fairValue - credit;
}

template <>
float Strategy<Side::Sell>::CalcPrice(float fairValue, float credit) {
  return fairValue + credit;
}

//// VERSION 2
template <Side> float CalcPrice(float fairValue, float credit);

template <> float CalcPrice<Side::Sell>(float fairValue, float credit) {
  return fairValue - credit;
}

template <> float CalcPrice<Side::Buy>(float fairValue, float credit) {
  return fairValue + credit;
}

void SendOrder(float orderPrice);
template <Side side> void RunStrategy(float fairValue, float credit) {
  float const orderPrice = CalcPrice<side>(fairValue, credit);
  SendOrder(orderPrice);
}

///////////////////////////////////////////////////
struct OrderSenderA {
  void SendOrder();
};
struct OrderSenderB {
  void SendOrder();
};

struct IOrderManager {
  virtual void MainLoop() = 0;
  virtual ~IOrderManager() = default;
};
template <typename T> struct OrderManager : public IOrderManager {
  void MainLoop() final { mOrderSender.SendOrder(); }
  T mOrderSender;
};

struct Config {
  bool UseOrderSenderA() const { return true; };
};

std::unique_ptr<IOrderManager> Factory(Config const &config) {
  if (config.UseOrderSenderA())
    return std::make_unique<OrderManager<OrderSenderA>>();
  else
    return std::make_unique<OrderManager<OrderSenderB>>();
}

int main() {
  auto manager = Factory(Config{});
  manager->MainLoop();

  RunStrategy<Side::Buy>(3.0, 7.3);
}
