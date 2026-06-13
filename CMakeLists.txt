cmake_minimum_required(VERSION 3.18)
project(quantcore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

pybind11_add_module(_core
    src/cpp/binding.cpp
    src/cpp/brownian.cpp
    src/cpp/gbm.cpp
    src/cpp/pricing.cpp
    src/cpp/markov.cpp
)

target_link_libraries(_core PRIVATE Threads::Threads)

install(TARGETS _core DESTINATION quantcore)
