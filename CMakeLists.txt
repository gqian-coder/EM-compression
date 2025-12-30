cmake_minimum_required(VERSION 3.19)

project(4dSTEM)
find_package(mgard REQUIRED)
find_package(hip REQUIRED)

add_definitions(-DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -Wfatal-errors -Wall -Wextra")

add_executable(mgard_4dStem mgard_4dStem.cpp)
target_link_libraries(mgard_4dStem mgard::mgard)
target_link_libraries(mgard_4dStem hip::host)
target_compile_definitions(mgard_4dStem PRIVATE MGARD_STEM_USE_HIP_EVENTS=1)
