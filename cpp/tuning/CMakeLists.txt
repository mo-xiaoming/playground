cmake_minimum_required (VERSION 3.15)
project(Tuning VERSION 1.0.0.0 LANGUAGES CXX)

include(cmake/First.cmake)

# CXX flags
include(cmake/CXXFlags.cmake)

# conan
include(cmake/Conan.cmake)

# dynamic tools
include(cmake/Sanitizers.cmake)

add_compile_options(${COMMON_FLAGS})

add_executable(pass_param pass_param.cpp)
target_link_libraries(pass_param CONAN_PKG::benchmark)

add_executable(pair_emplace pair_emplace.cpp)
target_link_libraries(pair_emplace CONAN_PKG::benchmark)

add_executable(prefetch prefetch.cpp)
target_link_libraries(prefetch CONAN_PKG::benchmark)

add_executable(n-set n-set.cpp)
target_include_directories(n-set SYSTEM PRIVATE ${CONAN_INCLUDE_DIRS_BOOST})
target_link_libraries(n-set CONAN_PKG::benchmark)

add_executable(inst_par inst_par.cpp)
target_include_directories(inst_par SYSTEM PRIVATE ${CONAN_INCLUDE_DIRS_BOOST})
target_link_libraries(inst_par CONAN_PKG::benchmark)

add_executable(crazy_world crazy_world.cpp)
target_include_directories(crazy_world SYSTEM PRIVATE ${CONAN_INCLUDE_DIRS_BOOST})
target_link_libraries(crazy_world CONAN_PKG::benchmark)

add_executable(move_iterator move_iterator.cpp)
target_link_libraries(move_iterator CONAN_PKG::benchmark)

add_executable(sso sso.cpp)
target_include_directories(sso SYSTEM PRIVATE ${CONAN_INCLUDE_DIRS_BOOST})
target_link_libraries(sso CONAN_PKG::benchmark)

add_executable(cache_size cache_size.cpp)
target_link_libraries(cache_size CONAN_PKG::benchmark)

add_executable(cache cache.cpp)
target_include_directories(cache SYSTEM PRIVATE ${CONAN_INCLUDE_DIRS_BOOST})
target_link_libraries(cache CONAN_PKG::benchmark)

add_executable(step step.cpp)
target_link_libraries(step CONAN_PKG::benchmark)

add_executable(cache_line cache_line.cpp)
target_link_libraries(cache_line CONAN_PKG::benchmark)

add_executable(pow pow.cpp)
target_link_libraries(pow CONAN_PKG::benchmark)

add_executable(mod mod.cpp)
target_include_directories(mod SYSTEM PRIVATE ${CONAN_INCLUDE_DIRS_BOOST})
target_link_libraries(mod CONAN_PKG::benchmark)

add_executable(align align.cpp)
target_link_libraries(align CONAN_PKG::benchmark)

add_executable(false_sharing false_sharing.cpp)
target_include_directories(false_sharing SYSTEM PRIVATE ${CONAN_INCLUDE_DIRS_BOOST})
target_link_libraries(false_sharing CONAN_PKG::benchmark)

add_executable(prediction prediction.cpp)
target_include_directories(prediction SYSTEM PRIVATE ${CONAN_INCLUDE_DIRS_BOOST})
target_link_libraries(prediction CONAN_PKG::benchmark)

add_executable(virtual virtual.cpp)
target_include_directories(virtual SYSTEM PRIVATE ${CONAN_INCLUDE_DIRS_BOOST})
target_link_libraries(virtual CONAN_PKG::benchmark)

add_executable(matrix matrix.cpp)
target_include_directories(matrix SYSTEM PRIVATE ${CONAN_INCLUDE_DIRS_BOOST})
target_link_libraries(matrix CONAN_PKG::benchmark)

add_executable(exception exception.cpp)
target_link_libraries(exception CONAN_PKG::benchmark)

add_executable(data_or data_or.cpp)
target_include_directories(data_or SYSTEM PRIVATE ${CONAN_INCLUDE_DIRS_BOOST})
target_link_libraries(data_or CONAN_PKG::benchmark)

add_executable(pass_to_vector_push pass_to_vector_push.cpp)
target_link_libraries(pass_to_vector_push CONAN_PKG::benchmark)

add_executable(vec_perf_benchmark vec_perf/vec_perf_benchmark.cpp)
target_link_libraries(vec_perf_benchmark CONAN_PKG::benchmark CONAN_PKG::fmt)

add_executable(find_perf vec_perf/find_perf.cpp)
target_link_libraries(find_perf CONAN_PKG::benchmark CONAN_PKG::fmt)

add_executable(vec vec_perf/vec.cpp)
target_include_directories(vec SYSTEM PRIVATE ${CONAN_INCLUDE_DIRS_BOOST})
target_link_libraries(vec CONAN_PKG::boost CONAN_PKG::fmt)
