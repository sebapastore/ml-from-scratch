#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <matrix.h>

/*
 * Linear Regression model
 *
 * y = XW + b
 *
 * - weights: (n_features x 1)
 * - bias: scalar
 */
typedef struct
{
    Matrix weights;
    float bias;
} LinearRegression;

/* =========================================================
 * Lifecycle
 * ========================================================= */

/* Initialize model with given number of features */
void lr_init(LinearRegression *lr, int n_features);

/* Free internal resources */
void lr_free(LinearRegression *lr);

/* =========================================================
 * Training
 * ========================================================= */

/*
 * Fit model using batch gradient descent
 *
 * X: (n_samples x n_features)
 * y: (n_samples x 1)
 */
void lr_fit(
    LinearRegression *lr,
    const Matrix *X,
    const Matrix *y,
    float learning_rate,
    int epochs);

/* =========================================================
 * Inference
 * ========================================================= */

/*
 * Predict outputs for input samples
 *
 * X:    (n_samples x n_features)
 * out:  (n_samples x 1)
 */
void lr_predict(
    Matrix *out,
    const LinearRegression *lr,
    const Matrix *X);

#endif
