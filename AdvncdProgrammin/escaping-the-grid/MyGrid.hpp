#pragma once

#include "Grid.hpp"

class MyGrid : public Grid {
public:

    // The Constructor
    size_t rows_; size_t cols_;
    const Tile& initial_tile;
    // Constructs a grid with the given size and initial tile.
    MyGrid(size_t rows, size_t cols, const Tile& initialTile): initial_tile(initialTile)
    {
        rows_ = rows;
        cols_ = cols;
    };

    // The Copy Constructor
    //
    // Constructs a grid with the same size and contents as the supplied
    // other grid.
    MyGrid(const MyGrid& grid): initial_tile(grid.initial_tile){
        
    };

    // The Move Constructor
    //
    // Constructs a grid with the same size and contents as tha supplied
    // other grid.  Potentially reuse data from that other grid.
    MyGrid(MyGrid&&) noexcept;

    // The Move Assignment Operator
    //
    //
    MyGrid& operator=(MyGrid&&) noexcept;

    // The remaining functions are those inherited from the abstract Grid
    // class.  We add the 'override' specifier to each of them to declare
    // our intent.  This way, the compiler can warn us when one of these
    // functions doesn't actually override a member function from the base
    // class, e.g., because we forgot a 'const' somewhere.

    ~MyGrid() override;

    MyGrid& operator=(const Grid&);

    size_t rows() const override{
        return rows_;
    };

    size_t cols() const override{
        return cols_;
    };

    bool validPosition(size_t row, size_t col) const noexcept override;

    Tile& operator()(size_t row, size_t col) override;

    const Tile& operator()(size_t row, size_t col) const override;

    void print(std::ostream&) const override;
    static MyGrid read(std::istream&);

protected:
    size_t current_row; size_t current_col;
};
