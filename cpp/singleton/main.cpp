#include "logger.hpp"

int main() {
    logger::info([] { return "level==Info"; });
    logger::set_level(logger::Level::Error);
    logger::info("level==Error");
    logger::set_level(logger::Level::Debug);
    logger::info("level==Debug");
}
