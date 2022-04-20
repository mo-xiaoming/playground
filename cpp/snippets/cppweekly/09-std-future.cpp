#include <algorithm>
#include <future>
#include <iostream>
#include <iterator>
#include <random>
#include <set>
#include <vector>

template <typename T>
struct Rand_gen {
    Rand_gen(T min_val, T max_val)
        : min_val_(min_val)
        , max_val_(max_val)
    {
    }
    T next() { return dis_(gen_); }

private:
    T min_val_ = T {};
    T max_val_ = T {};
    std::random_device rd_ {};
    std::mt19937 gen_ { rd_() };
    std::uniform_int_distribution<> dis_ { min_val_, max_val_ };
};

std::set<int> gen_nums1()
{
    auto gen = Rand_gen(0, 10'000'000);

    auto out = std::vector<int>();
    std::generate_n(std::back_inserter(out), 10'000'000, [&gen]() { return gen.next(); });
    return { out.cbegin(), out.cend() };
}

std::set<int> gen_nums()
{
    auto gen = Rand_gen(0, 10'000'000);

    auto out = std::set<int>();
    std::generate_n(std::inserter(out, out.end()), 10'000'000, [&gen]() { return gen.next(); });
    return out;
}

void version1()
{ // 34s
    std::cout << gen_nums1().size() << ' ' << gen_nums1().size() << '\n';
}

void version2()
{ // 29s
    std::cout << gen_nums().size() << ' ' << gen_nums().size() << '\n';
}

void version3()
{ // 29s
    std::cout << std::async(gen_nums).get().size() << ' ' << std::async(gen_nums).get().size() << '\n';
}

void version4()
{ // 29s
    std::cout << std::async(std::launch::async, gen_nums).get().size() << ' ' << std::async(std::launch::async, gen_nums).get().size() << '\n';
}

void version5()
{ // 29s
    auto f1 = std::async(std::launch::async, gen_nums);
    std::cout << f1.get().size() << ' ' << std::async(std::launch::async, gen_nums).get().size() << '\n';
}

void version6()
{ // 19s
    auto f2 = std::async(std::launch::async, gen_nums);
    std::cout << std::async(std::launch::async, gen_nums).get().size() << ' ' << f2.get().size() << '\n';
}

void version7()
{ // 20s
    auto f1 = std::async(std::launch::async, gen_nums);
    auto f2 = std::async(std::launch::async, gen_nums);
    std::cout << f1.get().size() << ' ' << f2.get().size() << '\n';
}

void version8()
{ // 20s
    auto f1 = std::async(gen_nums);
    auto f2 = std::async(gen_nums);
    std::cout << f1.get().size() << ' ' << f2.get().size() << '\n';
}

int main()
{
    version8();
}
