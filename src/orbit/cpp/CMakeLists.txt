cmake_minimum_required(VERSION 3.18)
project(orbit_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# locate pybind11 (also installs its cmake config so find_package works).
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_orbit_core
  src/bindings.cpp
  src/dag.cpp
  src/executor.cpp
  src/event_loop.cpp
  src/simulator.cpp
)

target_include_directories(_orbit_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# install the built extension inside the orbit package so it is importable as
# orbit._orbit_core
install(TARGETS _orbit_core LIBRARY DESTINATION orbit)
