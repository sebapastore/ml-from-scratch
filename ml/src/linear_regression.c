#include <assert.h>
#include <stdlib.h>

#include "linalg.h"
#include "linear_regression.h"

/* =========================================================
 * Lifecycle
 * ========================================================= */

void lr_init(LinearRegression *lr, int n_features)
{
    assert(lr != NULL);
    assert(n_features > 0);

    lr->weights = mat_create(n_features, 1);
    mat_fill(&lr->weights, 0.0f);
    lr->bias = 0.0f;
}

void lr_free(LinearRegression *lr)
{
    assert(lr != NULL);
    mat_free(&lr->weights);
}

/* =========================================================
 * Inference: ŷ = XW + b
 * ========================================================= */

void lr_predict(
    Matrix *out,
    const LinearRegression *lr,
    const Matrix *X)
{
    assert(out != NULL);
    assert(lr != NULL);
    assert(X != NULL);

    /* out = X * W */
    mat_mul(out, X, &lr->weights);

    /* out += bias */
    int total = out->rows * out->cols;
    for (int i = 0; i < total; ++i)
    {
        out->data[i] += lr->bias;
    }
}

/* =========================================================
 * Training (Batch Gradient Descent)
 * ŷ = XW + b
 * error = ŷ − y
 * ∇W = (1/N) Xᵀ error
 * ∇b = mean(error)
 * ========================================================= */

void lr_fit(
    LinearRegression *lr,
    const Matrix *X,
    const Matrix *y,
    float learning_rate,
    int epochs)
{
    assert(lr != NULL);
    assert(X != NULL);
    assert(y != NULL);
    assert(learning_rate > 0.0f);
    assert(epochs > 0);

    int n_samples = X->rows;
    int n_features = X->cols;

    /* Shape checks */
    assert(y->rows == n_samples);
    assert(y->cols == 1);
    assert(lr->weights.rows == n_features);

    /* Temporary matrices */
    Matrix y_pred = mat_create(X->rows, 1);
    Matrix error = mat_create(X->rows, 1);
    Matrix Xt = mat_create(X->cols, X->rows);
    Matrix grad_w = mat_create(n_features, 1);

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        /* y_pred = XW + b */
        lr_predict(&y_pred, lr, X);

        /* error = y_pred - y */
        mat_sub(&error, &y_pred, y);

        /* grad_w = (1/N) * X^T * error */
        mat_transpose(&Xt, X);
        mat_mul(&grad_w, &Xt, &error);
        mat_scale(&grad_w, &grad_w, 1.0f / n_samples);

        /* W = W - lr * grad_w */
        for (int i = 0; i < grad_w.cols; ++i)
        {
            lr->weights.data[i] -= learning_rate * grad_w.data[i];
        }

        /* bias gradient = mean(error) */
        float grad_b = mean(error);
        lr->bias -= learning_rate * grad_b;
    }

    /* Cleanup */
    mat_free(&y_pred);
    mat_free(&error);
    mat_free(&Xt);
    mat_free(&grad_w);
}
