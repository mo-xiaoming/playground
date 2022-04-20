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
    bool try_pop(std::function<void()>& f) {
        auto lock = std::unique_lock(mutex_, std::try_to_lock);
        if (! lock || q_.empty()) {
            return false;
        }
        f = std::move(q_.front());
        q_.pop_front();
        return true;
    }
    template <typename F>
    bool try_push(F&& f) {
        {
            auto lock = std::unique_lock(mutex_, std::try_to_lock);
            if (! lock) {
                return false;
            }
            q_.emplace_back(std::forward<F>(f));
        }
        ready_.notify_one();
        return true;
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
        for (auto n = 0U; n < count_; ++n) {
            threads_.emplace_back([this, n]{ run(n); });
        }
    }
    ~task_system() {
        for (auto& e: q_) {
            e.done();
        }
        for (auto& e: threads_) {
            e.join();
        }
    }
    template <typename F>
    void async_(F&& f) {
        auto const i = index_++;
        for (auto n = 0U; n < count_; ++n) {
            if (q_[(i+n) % count_].try_push(std::forward<F>(f))) {
                return;
            }
        }
        q_[i % count_].push(std::forward<F>(f));
    }
    void done() {
        for (auto &e : q_) {
            e.done();
        }
    }
private:
    void run(unsigned i) {
        while (true) {
            auto f = std::function<void()>();
            for (auto n = 0U; n < count_; ++n) {
                if (q_[(i+n) % count_].try_pop(f)) {
                    break;
                }
            }
            if (!f && !q_[i].pop(f)) {
                break;
            }
            f();
        }
    }
    std::atomic<unsigned> index_{0};
    unsigned const count_ = std::thread::hardware_concurrency();
    std::vector<std::thread> threads_;
    std::vector<notification_queue> q_{count_};
};

int main() {
    auto ts = task_system();
    ts.async_([]{ std::puts("hello"); });
    ts.async_([]{ std::puts("world"); });
    ts.async_([]{ std::puts("haha"); });

    using namespace std::chrono_literals;
    std::this_thread::sleep_for(1s);
}
