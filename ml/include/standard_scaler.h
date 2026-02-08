#ifndef STANDARD_SCALER_H
#define STANDARD_SCALER_H

#include "matrix.h"

typedef struct
{
    Matrix mean; // (1 × n_features)
    Matrix std;  // (1 × n_features)
} StandardScaler;

void scaler_fit(StandardScaler *scaler, const Matrix *X);
void scaler_transform(Matrix *out, const Matrix *X, const StandardScaler *scaler);
void scaler_fit_transform(Matrix *out, StandardScaler *scaler, const Matrix *X);
void scaler_free(StandardScaler *scaler);

#endif