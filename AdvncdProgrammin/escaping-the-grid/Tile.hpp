#pragma once

// This file defines the Grid Tiles that are used in this assignment.
//
// In a real word example, we'd introduce a hierarchy of different Grid
// classes with an interface, virtual member functions and so forth.  But
// for this first exercise, we can simply use an enumeration.

enum Tile {Wall, Floor, Path};

Tile tile_from_char(const char c);
char char_from_tile(const Tile& t);

