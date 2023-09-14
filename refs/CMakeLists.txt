cmake_minimum_required(VERSION 3.26)
project(stencil_ref)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_COLOR_DIAGNOSTICS ON)
find_package(fmt REQUIRED)

add_executable(stencil_ref stencil.cxx)
target_link_libraries(stencil_ref fmt::fmt)
