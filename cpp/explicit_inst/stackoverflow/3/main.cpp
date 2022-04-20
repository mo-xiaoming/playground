#include <iostream>

#include "mytemplate_interface.hpp"
#include "notmain.hpp"

int main() { std::cout << notmain() + MyTemplate<int>().f(1) << std::endl; }
