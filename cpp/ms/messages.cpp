#include <algorithm>

template <int TimeRange, int MaxMessages>
struct ThrottlingControlManager {
    [[nodiscard]] bool check() noexcept {
        auto const tp = current_time_point();
        auto const up = std::upper_bound(std::cbegin(time_points_), std::cend(time_points_), tp);
        auto const lp = std::lower_bound(std::cbegin(time_points_), std::cend(time_points_), tp - TimeRange);
        return up - lp > MaxMessages;
    }
    void send() noexcept {
        time_points_.push_back(current_time_point());
    }
private:
    std::deque<int> time_points_;
};

int main() {
    auto manager = ThrottlingControlManager<10, 1000>();
    auto c = manager.check();
}
