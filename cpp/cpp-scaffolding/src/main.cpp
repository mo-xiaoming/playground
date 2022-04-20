#include <csignal>
#include <spdlog/spdlog.h>

namespace {
volatile std::sig_atomic_t signal_received = 0; // NOLINT
void sig_handler(int signal) {
    if (signal == SIGINT) {
        spdlog::debug("received SIGINT");
        signal_received = signal;
    }
}
} // namespace

int main() {
    if (std::signal(SIGINT, sig_handler) == SIG_ERR) { // NOLINT
        spdlog::error("can't catch SIGINT");
    }
    spdlog::info("this is a cpp project template");
}
