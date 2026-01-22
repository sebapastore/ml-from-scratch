#include "linalg.h"

#include "assert.h"
#include "stddef.h"

/* =========================================================
 * Core arithmetic
 * ========================================================= */

void mat_add(Matrix *out, const Matrix *a, const Matrix *b)
{
    assert(out && a && b);
    mat_assert_same_shape(a, b);
    mat_assert_same_shape(a, out);

    int total = a->rows * a->cols;

    for (int i = 0; i < total; ++i)
    {
        out->data[i] = a->data[i] + b->data[i];
    }
}

void mat_sub(Matrix *out, const Matrix *a, const Matrix *b)
{
    assert(out && a && b);
    mat_assert_same_shape(a, b);
    mat_assert_same_shape(a, out);

    int total = a->rows * a->cols;

    for (int i = 0; i < total; ++i)
    {
        out->data[i] = a->data[i] - b->data[i];
    }
}

void mat_scale(Matrix *out, const Matrix *a, float scalar)
{
    assert(out && a);
    mat_assert_same_shape(a, out);

    int total = a->rows * a->cols;

    for (int i = 0; i < total; ++i)
    {
        out->data[i] = a->data[i] * scalar;
    }
}

void mat_mul(Matrix *out, const Matrix *a, const Matrix *b)
{
    assert(out && a && b);
    mat_assert_mul_shape(a, b);
    assert(out->rows == a->rows);
    assert(out->cols == b->cols);

    for (int i = 0; i < out->rows; ++i)
    {
        for (int j = 0; j < out->cols; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < a->cols; ++k)
            {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            out->data[i * out->cols + j] = sum;
        }
    }
}

/* =========================================================
 * Transformations
 * ========================================================= */

void mat_transpose(Matrix *out, const Matrix *m)
{
    assert(out && m);
    assert(out->rows == m->cols);
    assert(out->cols == m->rows);

    for (int i = 0; i < m->rows; ++i)
    {
        for (int j = 0; j < m->cols; ++j)
        {
            out->data[j * out->cols + i] = m->data[i * m->cols + j];
        }
    }
}

/* =========================================================
 * Reductions
 * ========================================================= */

float mat_sum(const Matrix *m)
{
    assert(m && m->data);

    float sum = 0.0f;
    int total = m->rows * m->cols;

    for (int i = 0; i < total; i++)
    {
        sum += m->data[i];
    }

    return sum;
}

float mat_mean(const Matrix *m)
{
    assert(m && m->data);
    assert(m->rows > 0 && m->cols > 0);

    int total = m->rows * m->cols;
    return mat_sum(m) / (float)total;
}

/* =========================================================
 * Elementwise operations
 * ========================================================= */

void mat_apply(Matrix *out, const Matrix *m, float (*fn)(float))
{
    assert(out && m);
    assert(fn);
    mat_assert_same_shape(out, m);

    int total = m->rows * m->cols;
    for (int i = 0; i < total; ++i)
    {
        out->data[i] = fn(m->data[i]);
    }
}