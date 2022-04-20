find_package(spdlog REQUIRED)
find_package(uwebsockets REQUIRED)

function(add_example name sources)
    add_executable(${name} ${sources})
    target_link_libraries(${name} PRIVATE spdlog::spdlog uwebsockets::uwebsockets fmt::fmt)
endfunction()

add_example(uwebsockets_basic_server uwebsockets/basic_server.cpp)
add_example(uwebsockets_basic_server_threaded uwebsockets/basic_server_threaded.cpp)
add_example(uwebsockets_echo_websocket_server uwebsockets/echo_websocket_server.cpp)
add_example(uwebsockets_echo_websocket_server_broadcasting uwebsockets/echo_websocket_server_broadcasting.cpp)
add_example(uwebsockets_echo_websocket_server_threaded uwebsockets/echo_websocket_server_threaded.cpp)
add_example(uwebsockets_backpressure uwebsockets/backpressure.cpp)

find_package(uvw REQUIRED)
add_executable(uvw_main uvw/main.cpp)
target_link_libraries(uvw_main PRIVATE spdlog::spdlog uvw::uvw)

find_package(libuv REQUIRED)
add_executable(libuv_main libuv/main.cpp libuv/main.cpp)
target_link_libraries(libuv_main PRIVATE spdlog::spdlog libuv::libuv)
