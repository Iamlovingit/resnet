CMAKE_MINIMUM_REQUIRED(VERSION 3.10.0)
project(resnet)

file(GLOB source src/*)


message(${source})

include_directories(/usr/local/include)
link_directories(/usr/local/lib)
link_libraries(gflags
        PocoFoundation)

add_executable(resnet ${source})

