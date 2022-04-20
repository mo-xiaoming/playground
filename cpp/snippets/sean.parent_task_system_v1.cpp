#include <deque>
#include <vector>
#include <functional>
#include <memory>
#include <mutex>
#include <condition_variable>

#include <cstdio>
#include <thread>

struct notification_queue {
    bool pop(std::function<void()>& f) {
        auto lock = std::unique_lock(mutex_);
        while (q_.empty() && ! done_) {
            ready_.wait(lock);
        }
        if (q_.empty()) {
            return false;
        }
        f = std::move(q_.front());
        q_.pop_front();
        return true;
    }
    template <typename F>
    void push(F&& f) {
        {
            auto const lock = std::lock_guard(mutex_);
            q_.emplace_back(std::forward<F>(f));
        }
        ready_.notify_one();
    }
    void done() {
        {
            auto const lock = std::lock_guard(mutex_);
            done_ = true;
        }
        ready_.notify_all();
    }
private:
    bool done_ = false;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::function<void()>> q_;
};

struct task_system {
    task_system() {
        for (auto n = 0U; n < std::thread::hardware_concurrency(); ++n) {
            threads_.emplace_back([this, n]{ run(n); });
        }
    }
    ~task_system() {
        done();
        for (auto& e: threads_) {
            e.join();
        }
    }
    template <typename F>
    void async_(F&& f) {
        q_.push(std::forward<F>(f));
    }
    void done() {
        q_.done();
    }
private:
    void run(unsigned) {
        while (true) {
            auto f = std::function<void()>();
            if (q_.pop(f)) {
                f();
            } else {
                break;
            }
        }
    }
    std::vector<std::thread> threads_;
    notification_queue q_;
};

int main() {
    auto ts = task_system();
    ts.async_([]{ std::puts("hello"); });
    ts.async_([]{ std::puts("world"); });

    using namespace std::chrono_literals;
    std::this_thread::sleep_for(1s);
}
