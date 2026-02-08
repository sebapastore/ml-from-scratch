#ifndef CSV_LOADER_H
#define CSV_LOADER_H

#include "matrix.h"

// Loads CSV file into X (features) and y (target)
// Assumes last column is target
int load_csv_to_matrix(const char *filename, Matrix *X, Matrix *y);

#endif
