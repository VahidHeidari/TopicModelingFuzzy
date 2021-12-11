cmake_minimum_required(VERSION 3.1)
project(FuzzyLDA)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17 -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++17 -Wall -Wextra -Wpedantic")

add_executable(gibbs_lda Gibbs-LDA.cpp)
add_executable(fuzzy_lda fuzzy-LDA.cpp)

add_executable(q_fuzzy_lda q-fuzzy-LDA.cpp)
add_executable(q_gibbs_lda q-Gibbs-LDA.cpp)

