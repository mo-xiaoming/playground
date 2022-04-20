find_package(spdlog REQUIRED)
find_package(opencv REQUIRED)
find_package(cxxopts REQUIRED)

add_executable(DisplayImage DisplayImage.cpp)
target_link_libraries(DisplayImage PRIVATE spdlog::spdlog opencv::opencv cxxopts::cxxopts)

add_executable(HelloWorld HelloWorld.cpp)
target_link_libraries(HelloWorld PRIVATE spdlog::spdlog opencv::opencv)

add_executable(ImageWatch ImageWatch.cpp)
target_link_libraries(ImageWatch PRIVATE spdlog::spdlog opencv::opencv cxxopts::cxxopts)

add_executable(TrackBar TrackBar.cpp)
target_link_libraries(TrackBar PRIVATE spdlog::spdlog opencv::opencv)

add_executable(Capture Capture.cpp)
target_link_libraries(Capture PRIVATE spdlog::spdlog opencv::opencv)