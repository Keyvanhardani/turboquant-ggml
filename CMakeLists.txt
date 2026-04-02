cmake_minimum_required(VERSION 3.16)
project(turboquant-ggml VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ─── Library ──────────────────────────────────────────────────────────

add_library(turboquant STATIC
    src/turboquant.c
    src/turboquant-tables.c
)

target_include_directories(turboquant PUBLIC include)
target_link_libraries(turboquant PRIVATE m)

# ─── Tests ────────────────────────────────────────────────────────────

enable_testing()

add_executable(test_turboquant tests/test_turboquant.c)
target_link_libraries(test_turboquant PRIVATE turboquant m)

add_test(NAME turboquant_tests COMMAND test_turboquant)

add_executable(test_full tests/test_full.c)
target_link_libraries(test_full PRIVATE turboquant m)
add_test(NAME turboquant_full COMMAND test_full)

# ─── Benchmark (optional) ─────────────────────────────────────────────

option(TQ_BUILD_BENCH "Build benchmark" OFF)

if(TQ_BUILD_BENCH)
    add_executable(bench_turboquant tests/bench_turboquant.c)
    target_link_libraries(bench_turboquant PRIVATE turboquant m)
endif()

# ─── Install ──────────────────────────────────────────────────────────

install(TARGETS turboquant ARCHIVE DESTINATION lib)
install(FILES include/turboquant.h DESTINATION include)
