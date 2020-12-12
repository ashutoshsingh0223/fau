#include <cstddef>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <algorithm>
#include <functional>
#include <cassert>
#include <cmath>

#include "testing.hpp"
#include "MyGrid.hpp"

size_t count_neighboring_paths(const Grid& grid, size_t row, size_t col) {
    size_t count = 0;
    if (grid.validPosition(row+1,col) && (grid(row+1, col) == Path)) ++count;
    if (grid.validPosition(row-1,col) && (grid(row-1, col) == Path)) ++count;
    if (grid.validPosition(row,col+1) && (grid(row, col+1) == Path)) ++count;
    if (grid.validPosition(row,col-1) && (grid(row, col-1) == Path)) ++count;
    return count;
}

void test_escape(std::istream& stream) {
    MyGrid grid = MyGrid::read(stream);
    size_t rows = grid.rows();
    size_t cols = grid.cols();

    MyGrid copy(grid);
    copy.escape();
    assert(copy.rows() == grid.rows());
    assert(copy.cols() == grid.cols());
    assert(copy(1,1) == Path);
    bool path_reaches_boundary = false;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            switch (copy(row,col)) {
            case Wall:
                assert(grid(row,col) == Wall);
                break;
            case Floor:
                assert(grid(row,col) == Floor);
                break;
            case Path:
                assert(grid(row,col) == Floor);
                // Count neighboring Path tiles.
                if ((row == 0) || (col == 0) || (row == rows-1) || (col == cols-1)) {
                    // Boundary tiles.
                    assert(count_neighboring_paths(copy, row, col) == 1);
                    path_reaches_boundary = true;
                } else if ((row == 1) && (col == 1)) {
                    // The initial tile.
                    assert(count_neighboring_paths(copy, row, col) == 1);
                } else {
                    // Other tiles.
                    assert(count_neighboring_paths(copy, row, col) == 2);
                }
                break;
            }
        }
    }
    assert(path_reaches_boundary);
}

void test_escape_A() {
    TESTCASE(test_escape_A);
    std::ifstream file("../grids/A");
    test_escape(file);
}

void test_escape_B() {
    TESTCASE(test_escape_B);
    std::ifstream file("../grids/B");
    test_escape(file);
}

void test_escape_C() {
    TESTCASE(test_escape_C);
    std::ifstream file("../grids/C");
    test_escape(file);
}

void test_escape_D() {
    TESTCASE(test_escape_D);
    std::ifstream file("../grids/D");
    test_escape(file);
}

int main(int argc, char** argv) {
    test_escape_A();
    test_escape_B();
    test_escape_C();
    test_escape_D();
}
