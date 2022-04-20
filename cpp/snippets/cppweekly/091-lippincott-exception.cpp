#include <iostream>

extern void bar1();
extern void bar2();

void lippincott() {
    try {
        throw;
    } catch(std::runtime_error &e) {
        std::clog << e.what();
    } catch(std::exception &e)  {
        std::clog << e.what();
    } catch(...) {
        std::clog << "unknown";
    }
}

void foo1() {
    try {
        bar1();
    } catch(...) {
        lippincott();
    }
}

void foo2() {
    try {
        bar2();
    } catch(...) {
        lippincott();
    }
}
