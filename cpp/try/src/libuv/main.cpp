#include <spdlog/spdlog.h>
#include <uv.h>

static void callback(uv_timer_t* /*handle*/) { spdlog::info("timer fired"); }

int main() {
    uv_loop_t* loop = uv_default_loop();

    uv_timer_t timer;

    uv_timer_init(loop, &timer);
    uv_timer_start(&timer, callback, 0L, 2000L);

    return uv_run(loop, UV_RUN_DEFAULT);
}
