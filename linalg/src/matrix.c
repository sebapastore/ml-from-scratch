#include "matrix.h"

#include "stdlib.h"
#include "assert.h"
#include "string.h"

/* =========================================================
 * Memory & lifecycle
 * ========================================================= */

Matrix mat_create(int rows, int cols)
{
    assert(rows > 0);
    assert(cols > 0);

    Matrix m;

    m.rows = rows;
    m.cols = cols;
    m.data = malloc((size_t)rows * (size_t)cols * sizeof(float));

    assert(m.data != NULL);

    return m;
}

void mat_free(Matrix *m)
{
    if (!m || !m->data)
    {
        return;
    }

    free(m->data);
    m->data = NULL;
    m->rows = 0;
    m->cols = 0;
}

/* =========================================================
 * Element access
 * ========================================================= */

static inline size_t mat_index(const Matrix *m, int row, int col)
{
    return (size_t)row * (size_t)m->cols + (size_t)col;
}

float mat_get(const Matrix *m, int row, int col)
{
    assert(m);
    assert(m->data);
    assert(row >= 0 && row < m->rows);
    assert(col >= 0 && col < m->cols);

    return m->data[mat_index(m, row, col)];
}

void mat_set(Matrix *m, int row, int col, float value)
{
    assert(m);
    assert(m->data);
    assert(row >= 0 && row < m->rows);
    assert(col >= 0 && col < m->cols);

    m->data[mat_index(m, row, col)] = value;
}

/* =========================================================
 * Initialization / copy
 * ========================================================= */

void mat_fill(Matrix *m, float value)
{
    assert(m);
    assert(m->data);

    int total = m->rows * m->cols;

    for (int i = 0; i < total; ++i)
    {
        m->data[i] = value;
    }
}

void mat_copy(Matrix *dst, const Matrix *src)
{
    assert(src);
    assert(src->data);
    assert(dst);
    assert(dst->data);
    assert(dst->rows == src->rows);
    assert(dst->cols == src->cols);

    size_t bytes = (size_t)src->rows * (size_t)src->cols * sizeof(float);
    memcpy(dst->data, src->data, bytes);
}

/* =========================================================
 * Shape assertions
 * ========================================================= */

void mat_assert_same_shape(const Matrix *a, const Matrix *b)
{
    assert(a);
    assert(b);
    assert(a->rows == b->rows);
    assert(a->cols == b->cols);
}

void mat_assert_mul_shape(const Matrix *a, const Matrix *b)
{
    assert(a);
    assert(b);
    assert(a->cols == b->rows);
}
