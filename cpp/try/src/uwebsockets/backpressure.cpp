#include <spdlog/spdlog.h>
#include <uWebSockets/App.h>

int main() {
    struct Data {};

    constexpr auto backpressure = 1024;
    auto message_number = 0;
    auto messages = 0;
    auto const f = [&message_number, &messages, backpressure](auto* ws, std::string_view caller) {
        return [&messages, &message_number, backpressure, caller](auto* ws) {
            while (ws->getBufferedAmount() < backpressure) {
                ws->send(fmt::format("{}: let's call it {}", caller, message_number), uWS::TEXT, true);
                ++message_number;
                ++messages;
            }
        };
    };

    auto behavior = uWS::App::WebSocketBehavior{
        .compression = uWS::SHARED_COMPRESSOR,
        .open = [f](auto* ws) { return f(ws, "open"); },
        .drain = [f](auto* ws) { return f(ws, "drain"); },
    };

    constexpr auto port = 9001;
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
