#include "Grid.hpp"

std::ostream& operator<<(std::ostream& out, const Grid& grid) {
    grid.print(out);
    return out;
}
