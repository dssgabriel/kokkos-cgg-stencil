cmake_minimum_required(VERSION 3.26)
project(kokkos_scratch)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_COLOR_DIAGNOSTICS ON)
find_package(fmt REQUIRED)
add_subdirectory(ext/kokkos)

add_executable(kokkos_scratch src/main.cpp)
target_link_libraries(kokkos_scratch fmt::fmt)
target_link_libraries(kokkos_scratch Kokkos::kokkos)
