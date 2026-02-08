#include <assert.h>
#include <math.h>

#include "standard_scaler.h"
#include "linalg.h"

void scaler_fit(StandardScaler *scaler, const Matrix *X)
{
    assert(X);
    assert(X->rows > 0);
    assert(X->cols > 0);

    scaler->mean = mat_create(1, X->cols);
    scaler->std = mat_create(1, X->cols);

    for (int j = 0; j < X->cols; j++)
    {
        float mean = mat_col_mean(X, j);
        float std = mat_col_std(X, j);

        mat_set(&scaler->mean, 0, j, mean);
        mat_set(&scaler->std, 0, j, std);
    }
}

void scaler_transform(Matrix *out, const Matrix *X, const StandardScaler *scaler)
{
    assert(out);
    assert(X);
    assert(scaler);
    assert(X->cols == scaler->mean.cols);
    assert(X->cols == scaler->std.cols);

    *out = mat_create(X->rows, X->cols);

    for (int i = 0; i < X->rows; i++)
    {
        for (int j = 0; j < X->cols; j++)
        {
            float x = mat_get(X, i, j);
            float mean = mat_get(&scaler->mean, 0, j);
            float std = mat_get(&scaler->std, 0, j);

            mat_set(out, i, j, (x - mean) / std);
        }
    }
}

void scaler_fit_transform(Matrix *out, StandardScaler *scaler, const Matrix *X)
{
    scaler_fit(scaler, X);
    scaler_transform(out, X, scaler);
}

void scaler_free(StandardScaler *scaler)
{
    mat_free(&scaler->mean);
    mat_free(&scaler->std);
}
