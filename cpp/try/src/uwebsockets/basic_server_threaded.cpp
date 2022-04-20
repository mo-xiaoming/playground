#include <algorithm>
#include <thread>

#include <fmt/ostream.h>
#include <spdlog/spdlog.h>
#include <uWebSockets/App.h>

int main() {

    auto threads = std::vector<std::thread>(std::thread::hardware_concurrency());
    constexpr int port = 9001;
    auto const handler = [port] {
        uWS::App()
            .get("/hello",
                 [](auto* res, auto* /*req*/) {
                     spdlog::info(fmt::format("{} got the request", std::this_thread::get_id()));
                     res->writeHeader("Content-Type", "text/html; charset=utf-8")->end("Hello HTTP!");
                 })
            .listen(port,
                    [port](auto* listenSocket) {
                        if (listenSocket) {
                            spdlog::info(fmt::format(FMT_STRING("{} is listening on port {}"),
                                                     std::this_thread::get_id(), port));
                        } else {
                            spdlog::error(fmt::format(FMT_STRING("{} faile to listen on port {}"),
                                                      std::this_thread::get_id(), port));
                        }
                    })
            .run();
    };

    std::generate_n(threads.begin(), threads.size(), [handler] { return std::thread(handler); });
    for (auto& t : threads) {
        t.join();
    }

    spdlog::critical("Failed to listen on port {}", port);
}
