#include <cstring>

int main() {
    auto const a = R"something(haha)something";
    return std::strlen(a) == std::strlen("haha");
}

