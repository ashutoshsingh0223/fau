project(AdvPT)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -pedantic")

add_executable( gridtest Tile.cpp Grid.cpp MyGrid.cpp gridtest.cpp)

add_executable( escapetest Tile.cpp Grid.cpp MyGrid.cpp escape.cpp escapetest.cpp)
