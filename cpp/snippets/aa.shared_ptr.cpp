template <typename T>
struct Single_thread_ptr0 {
    explicit Single_thread_ptr0(T* p = nullptr) : p_(p), c_(p ? new int(1) : nullptr) {}
    Single_thread_ptr0(Single_thread_ptr0 const& rhs) : p_(rhs.p_), c_(rhs.c_) {
        if (c_) {
            ++(*c_);
        }
    }
    Single_thread_ptr0(Single_thread_ptr0&& rhs) : p_(rhs.p_), c_(rhs.c_) {
        rhs.p_ = nullptr;
        rhs.c_ = nullptr;
    }
    ~Single_thread_ptr0() {
        if (c_ && --(*c_) == 0) {
            delete p_;
            delete c_;
        }
    }
private:
    T* p_ = nullptr;
    int* c_ = nullptr;
};


// One ref: p_ && (c_ == nullptr || *c_ == 1)
// Many refs: p_ && c_ && *c_ > 1
template <typename T>
struct Single_thread_ptr1 {
    explicit Single_thread_ptr1(T* p = nullptr) : p_(p), c_(nullptr) {}
    Single_thread_ptr1(Single_thread_ptr1 const& rhs) : p_(rhs.p_), c_(rhs.c_) {
        if (p_ == nullptr) {
            return;
        }
        if (c_ == nullptr) {
            c_ = rhs.c_ = new int(2);
        } else {
            ++(*c_);
        }
    }
    Single_thread_ptr1(Single_thread_ptr1&& rhs) : p_(rhs.p_), c_(rhs.c_) {
        rhs.p_ = nullptr;
        rhs.c_ = nullptr;
    }
    ~Single_thread_ptr1() {
        if (c_ == nullptr) {
            soSueMe: delete p_;
        } else if (--(*c_) == 0) {
            delete c_;
            goto soSueMe;
        }
    }
private:
    T* p_ = nullptr;
    int* c_ = nullptr;
};

template <typename T>
struct Single_thread_ptr2 {
    explicit Single_thread_ptr2(T* p = nullptr) : p_(p), c_(nullptr) {}
    Single_thread_ptr2(Single_thread_ptr2 const& rhs) : p_(rhs.p_), c_(rhs.c_) {
        if (p_ == nullptr) {
            return;
        }
        if (c_ == nullptr) {
            c_ = rhs.c_ = new int(2);
        } else {
            ++(*c_);
        }
    }
    Single_thread_ptr2(Single_thread_ptr2&& rhs) : p_(rhs.p_), c_(rhs.c_) {
        rhs.p_ = nullptr;
        // rhs.c_ = nullptr; // UNNEEDED
    }
    ~Single_thread_ptr2() {
        if (p_ == nullptr) {
            return;
        }
        if (c_ == nullptr) {
            soSueMe: delete p_;
        } else if (*c_ == 1) { // avoid writing
            delete c_;
            goto soSueMe;
        } else {
            --(*c_);
        }
    }
private:
    T* p_ = nullptr;
    int* c_ = nullptr;
};
