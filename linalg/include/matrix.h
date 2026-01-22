#ifndef MATRIX_H
#define MATRIX_H

/* =========================================================
 * Row-major Matrix
 * ========================================================= */
typedef struct
{
    int rows, cols;
    float *data;
} Matrix;

/* =========================================================
 * Memory & lifecycle
 * ========================================================= */

Matrix mat_create(int rows, int cols);
void mat_free(Matrix *m);

/* =========================================================
 * Element access
 * ========================================================= */

float mat_get(const Matrix *m, int row, int col);
void mat_set(Matrix *m, int row, int col, float value);

/* =========================================================
 * Initialization / copy
 * ========================================================= */

void mat_fill(Matrix *m, float value);
void mat_copy(Matrix *dst, const Matrix *src);

/* =========================================================
 * Shape checks
 * ========================================================= */

void mat_assert_same_shape(const Matrix *a, const Matrix *b);
void mat_assert_mul_shape(const Matrix *a, const Matrix *b);

#endif