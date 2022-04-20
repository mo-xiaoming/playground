#include <iostream>

struct B {
    virtual void foo() const { std::cout << "B::foo\n"; }
    virtual ~B() { std::cout << "~B\n"; }
    virtual void raise() const { throw *this; }
};

struct D : B {
    void foo() const override { std::cout << "D::foo\n"; }
    ~D() { std::cout << "~D\n"; }
    void raise() const override { throw *this; }
};

void foo(B const& b) {
    b.raise();
}

int main() {
    try {
        foo(D());
    } catch (B const& b) {
        b.foo();
    }
}
