#include <algorithm>
#include <thread>

#include <fmt/ostream.h>
#include <spdlog/spdlog.h>
#include <uWebSockets/App.h>

int main() {
    struct Data {
        int i = 0;
    };

    auto const make_behavior = [] {
        return uWS::App::WebSocketBehavior{
            .compression = uWS::SHARED_COMPRESSOR,
            .open =
                [](auto* ws) {
                    spdlog::info("{} open: {:p}", std::this_thread::get_id(), fmt::ptr(ws->getUserData()));
                },
            .message =
                [](auto* ws, std::string_view message, uWS::OpCode code) {
                    auto* data = static_cast<Data*>(ws->getUserData());
                    spdlog::info("message {}: code {} message {}", data->i, code, message);
                    data->i++;
                    ws->send(fmt::to_string(data->i), code, true);
                },
            .drain = [](auto* ws) { spdlog::info("drain: {:p}", fmt::ptr(ws->getUserData())); },
            .ping = [](auto* /*ws*/) { spdlog::info("ping"); },
            .pong = [](auto* /*ws*/) { spdlog::info("pong"); },
            .close =
                [](auto* ws, int code, std::string_view message) {
                    spdlog::info("close: code {} message {} {:p}", code, message, fmt::ptr(ws->getUserData()));
                },
        };
    };

    auto const handler = [make_behavior] {
        constexpr int port = 9001;
        uWS::App()
            .ws<Data>("/*", make_behavior())
            .listen(port,
                    [port](auto* token) {
                        if (token) {
                            spdlog::info("Listening on port {}", port);
                        }
                    })
            .run();
    };

    auto pool = std::vector<std::thread>(std::thread::hardware_concurrency());
    std::generate(pool.begin(), pool.end(), [handler] { return std::thread(handler); });
    for (auto& t : pool) {
        t.join();
    }
}
