CMAKE_MINIMUM_REQUIRED(VERSION 3.10.0)
project(resnet)
set(CMAKE_CXX_STANDARD 11)
file(GLOB source src/*)


message(${source})

include_directories(/usr/local/include)
link_directories(/usr/local/lib)
link_libraries(gflags
        PocoFoundation)

add_executable(resnet ${source})

