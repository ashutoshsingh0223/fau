#include <cstddef>
#include <cassert>

// Feel free to include additional headers here!

#include "Grid.hpp"

// Feel free to define auxiliary functions here!

void Grid::escape() {
    Grid& grid = *this;
    assert(grid(1,1) == Floor); // Check that the initial tile is valid.

    // TODO implement some path finding algorithm find a correct path to an
    // exit tile, and then write that path to the grid.

}
