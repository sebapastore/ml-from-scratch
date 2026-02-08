#ifndef LINALG_H
#define LINALG_H

#include "matrix.h"

/* =========================================================
 * Core arithmetic
 * ========================================================= */

void mat_add(Matrix *out, const Matrix *a, const Matrix *b);
void mat_sub(Matrix *out, const Matrix *a, const Matrix *b);
void mat_scale(Matrix *out, const Matrix *a, float scalar);
void mat_mul(Matrix *out, const Matrix *a, const Matrix *b);

/* =========================================================
 * Transformations
 * ========================================================= */

void mat_transpose(Matrix *out, const Matrix *m);

/* =========================================================
 * Reductions
 * ========================================================= */

float mat_sum(const Matrix *m);
float mat_mean(const Matrix *m);

/* =========================================================
 * Elementwise operations
 * ========================================================= */

void mat_apply(Matrix *out, const Matrix *m, float (*fn)(float));

/* =========================================================
 * Column-wise statistics
 * ========================================================= */

float mat_col_mean(const Matrix *X, int col);
float mat_col_std(const Matrix *X, int col);
float mat_col_min(const Matrix *X, int col);
float mat_col_max(const Matrix *X, int col);

#endif