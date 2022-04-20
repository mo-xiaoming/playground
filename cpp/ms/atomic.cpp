#include <cassert>


template <typename T>
struct Shared_ptr {
    explicit Shared_ptr(T * p=nullptr): p_(p) {
        if (p_ != nullptr) {
            ref_ = new int(1);
        }
    }
    Shared_ptr(Shared_ptr const& other) {
        create(other);
    }
    Shared_ptr& operator=(Shared_ptr const& other) & {
        if (this == &other) {
            return *this;
        }
        destroy();
        create(other);
        return *this;
    }
    ~Shared_ptr() {
        destroy();
    }

    [[nodiscard]] T* get() const noexcept { return p_; }

    [[nodiscard]] int ref_count() const noexcept {
        if (ref_ != nullptr) {
            return *ref_;
        }
        return 0;
    }
private:
    T *p_ = nullptr;
    int *ref_ = nullptr;

    void create(Shared_ptr const& other)  {
        ref_ = other.ref_;
        if (ref_ != nullptr) {
            ++(*ref_);
        }
        p_ = other.p_;
    }
    void destroy() noexcept {
        if (ref_ == nullptr) {
            return;
        }
        if (ref_ != nullptr && *ref_ == 1) {
            delete p_;
            p_ = nullptr;
            delete ref_;
            ref_ = nullptr;
        } else {
            --(*ref_);
        }
    }
};

int main() {
    auto p0 = Shared_ptr<int>();
    assert(p0.get() == nullptr);
    assert(p0.ref_count() == 0);
    auto n = p0;
    assert(p0.get() == nullptr);
    assert(n.get() == nullptr);
    assert(p0.ref_count() == 0);
    assert(n.ref_count() == 0);


    auto *i = new int();
    auto p = Shared_ptr(i);
    assert(i == p.get());
    assert(p.ref_count() == 1);

    {
        auto q = p;
        assert(p.ref_count() == 2);
        assert(q.ref_count() == 2);
    }
    assert(p.ref_count() == 1);

    auto m = Shared_ptr(new int(3));
    m = p;
    assert(m.get() == p.get());
    assert(m.ref_count() == 2);
    assert(p.ref_count() == 2);
}

