#pragma once

#include <stdexcept>
#include <cstddef>
#include <iostream>
#include "Tile.hpp"

// This file defines an abstract base class for a two dimensional grid.  It
// only has pure virtual member functions, so any actual grid class that
// inherits from it has to implement all member functions in a sensible
// way.

class Grid {
public:

    // Note that an abstract class has no constructors - the whole point of
    // an abstract class is that it cannot be instantiated.

    // The Destructor
    //
    // Frees all resources associated with this grid.
    virtual ~Grid() = default;

    // The Copy Assignment Operator
    //
    // Overwrites the size and contents of a grid with those of another
    // supplied grid.
    virtual Grid& operator=(const Grid&) = 0;

    // The Grid Size Access Functions
    //
    // These functions should return the grid size that have been passed to
    // the constructor.
    virtual size_t rows() const = 0;
    virtual size_t cols() const = 0;

    // The Index Checker
    //
    // Returns true when the supplied indices reference a valid grid
    // element, and false otherwise.
    virtual bool validPosition(size_t row, size_t col) const noexcept = 0;

    // The Element Access Function
    //
    // Returns a constant reference to the tile at position row, col of the
    // grid.  Throws an invalid_grid_position exception if the specified
    // grid position is not valid.
    virtual const Tile& operator()(size_t row, size_t col) const = 0;

    // The Element Mutation Function
    //
    // Returns a mutable reference to the tile at position row, col of the
    // grid.  Throws an invalid_grid_position exception if the specified
    // grid position is not valid.
    virtual Tile& operator()(size_t row, size_t col) = 0;

    // The Printer
    //
    // Prints a Grid to the supplied output stream.
    virtual void print(std::ostream&) const = 0;

    // The escape functions used in part b) of the assignment.
    void escape();

    // Declare std::ostreams << operator as 'friend'.  Otherwise, its
    // implementation wouldn't be allowed to access the protected functions
    // of the Grid class.

    friend std::ostream& operator<<(std::ostream& out, const Grid& grid);
};

// Throw this exception we throw when someone attempts to reference a tile
// that doesn't exist.
class invalid_grid_position : public std::exception {
    virtual const char* what() const throw() {
        return "Invalid grid position.";
    }
};
