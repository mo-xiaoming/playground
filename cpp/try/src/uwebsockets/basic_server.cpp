#include <unistd.h>

#include <spdlog/spdlog.h>
#include <uWebSockets/App.h>

int main() {
    using namespace std::literals;

    constexpr int port = 9001;
    us_listen_socket_t* listen_token = nullptr;

    uWS::App()
        .get("/hello",
             [](auto* res, auto* /*req*/) {
                 auto hostname = std::array<char, HOST_NAME_MAX>();
                 ::gethostname(hostname.data(), hostname.size());
                 res->writeHeader("Content-Type", "text/html; charset=utf-8")->end("v2: "s + hostname.data());
             })
        .any("/done",
             [&listen_token](auto* res, auto* /*req*/) {
                 us_listen_socket_close(0, listen_token);
                 listen_token = nullptr;
                 res->end("graceful termination");
             })
        .any("/kill",
             [&listen_token](auto* res, auto* /*req*/) {
                 us_listen_socket_close(0, listen_token);
                 res->end("kill");
             })
        .listen(port,
                [port, &listen_token](auto* token) {
                    if (token) {
                        listen_token = token;
                        spdlog::info("Listening on port {}", port);
                    }
                })
        .run();
}
