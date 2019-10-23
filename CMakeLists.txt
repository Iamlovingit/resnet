CMAKE_MINIMUM_REQUIRED(VERSION 3.10.0)
project(resnet)

file(GLOB source src/*)

message(${source})

add_executable(resnet ${source})

