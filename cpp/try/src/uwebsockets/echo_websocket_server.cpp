#include <spdlog/spdlog.h>
#include <uWebSockets/App.h>

int main() {
    struct Data {
        int i = 0;
    };

    constexpr int port = 9001;
    auto behavior = uWS::App::WebSocketBehavior{};
    behavior.compression = uWS::SHARED_COMPRESSOR;
    behavior.open = [](auto* ws) { spdlog::info("open: {:p}", fmt::ptr(ws->getUserData())); };
    behavior.message = [](auto* ws, std::string_view message, uWS::OpCode code) {
        auto* data = static_cast<Data*>(ws->getUserData());
        spdlog::info("message {}: code {} message {}", data->i, code, message);
        data->i++;
        ws->send(fmt::to_string(data->i), code, true);
    };
    behavior.drain = [](auto* ws) { spdlog::info("drain: {:p}", fmt::ptr(ws->getUserData())); };
    behavior.ping = [](auto* /*ws*/) { spdlog::info("ping"); };
    behavior.pong = [](auto* /*ws*/) { spdlog::info("pong"); };
    behavior.close = [](auto* ws, int code, std::string_view message) {
        spdlog::info("close: code {} message {} {:p}", code, message, fmt::ptr(ws->getUserData()));
    };

    uWS::App()
        .ws<Data>("/*", std::move(behavior))
        .listen(port,
                [port](auto* token) {
                    if (token) {
                        spdlog::info("Listening on port {}", port);
                    }
                })
        .run();
}
