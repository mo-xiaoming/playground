#include <spdlog/spdlog.h>
#include <uWebSockets/App.h>

namespace fmt {
template <> struct [[maybe_unused]] formatter<uWS::OpCode> : formatter<std::string_view> {
    template <typename Format_context> auto format(uWS::OpCode code, Format_context& ctx) {
        auto const s = [code] {
            switch (code) {
            case uWS::TEXT:
                return "TEXT";
            case uWS::BINARY:
                return "BINARY";
            case uWS::CLOSE:
                return "CLOSE";
            case uWS::PING:
                return "PING";
            case uWS::PONG:
                return "PONG";
            }
            return "UNKNOWN";
        }();
        return formatter<std::string_view>::format(s, ctx);
    }
};
} // namespace fmt

int main() {
    struct Data {};

    us_listen_socket_t* listen_socket = nullptr;

    auto const make_behavior = [listen_socket] {
        return uWS::App::WebSocketBehavior{
            .compression = uWS::SHARED_COMPRESSOR,
            .open =
                [](auto* ws) {
                    spdlog::info("open: {:p}", fmt::ptr(ws->getUserData()));
                    ws->subscribe("broadcast");
                    spdlog::info("subscribed to broadcast");
                },
            .message =
                [listen_socket](auto* ws, std::string_view message, uWS::OpCode code) {
                    spdlog::info("code {} message {}", message, code, message);
                    if (message == "closedown") {
                        us_listen_socket_close(0, listen_socket);
                        ws->close();
                    }
                    ws->publish("broadcase", message, code, true);
                },
        };
    };

    constexpr auto port = 9001;
    uWS::App()
        .ws<Data>("/*", make_behavior())
        .listen(port,
                [port, &listen_socket](auto* token) {
                    if (token) {
                        listen_socket = token;
                        spdlog::info("Listening on port {}", port);
                    }
                })
        .run();
}
