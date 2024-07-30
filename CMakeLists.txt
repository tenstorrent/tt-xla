cmake_minimum_required(VERSION 3.20)


project(TT_PJRT)
cmake_policy(SET CMP0069 NEW)
set(TTPJRT_SOURCE_DIR ${PROJECT_SOURCE_DIR})

set(CMAKE_C_COMPILER clang-17)
set(DCMAKE_CXX_COMPILER clang++-17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti") 

# set(ASAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_FLAGS}")

link_directories(/opt/ttmlir-toolchain/lib/)

add_subdirectory(src)
add_subdirectory(third_party)
