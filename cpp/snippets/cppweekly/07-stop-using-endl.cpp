#include <iostream>

/* 17224 with '\n'
 * 17056 with "\n"
 * interesting
 */
int main(int, char** argv) {
    std::cout << argv[0] << "\n";
}
