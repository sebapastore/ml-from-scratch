#include "stdio.h"
#include "stdlib.h"
#include "linalg.h"
#include "linear_regression.h"

int main(void)
{
    // -------------------------------
    // Synthetic dataset: y = 2x + 1
    // -------------------------------
    int n_samples = 4;
    int n_features = 1;

    Matrix X = mat_create(n_samples, n_features);
    Matrix y = mat_create(n_samples, 1);

    // Fill X
    mat_set(&X, 0, 0, 0.0f);
    mat_set(&X, 1, 0, 1.0f);
    mat_set(&X, 2, 0, 2.0f);
    mat_set(&X, 3, 0, 3.0f);

    // Fill y
    mat_set(&y, 0, 0, 1.0f);
    mat_set(&y, 1, 0, 3.0f);
    mat_set(&y, 2, 0, 5.0f);
    mat_set(&y, 3, 0, 7.0f);

    // -------------------------------
    // Initialize Linear Regression
    // -------------------------------
    LinearRegression lr;
    lr_init(&lr, n_features);

    // Training parameters
    float learning_rate = 0.1f;
    int epochs = 100;

    // Temporary output for predictions
    Matrix y_pred = mat_create(n_samples, 1);

    // -------------------------------
    // Training loop (prints every 10 epochs)
    // -------------------------------
    for (int epoch = 1; epoch <= epochs; ++epoch)
    {
        lr_fit(&lr, &X, &y, learning_rate, 1); // fit one epoch at a time
        lr_predict(&y_pred, &lr, &X);

        // Compute MSE manually
        float mse = 0.0f;
        for (int i = 0; i < n_samples; ++i)
        {
            float diff = mat_get(&y_pred, i, 0) - mat_get(&y, i, 0);
            mse += diff * diff;
        }
        mse /= n_samples;

        if (epoch % 10 == 0 || epoch == 1 || epoch == epochs)
        {
            printf("Epoch %3d | MSE: %.4f | Weights: %.4f | Bias: %.4f\n",
                   epoch, mse, mat_get(&lr.weights, 0, 0), lr.bias);
        }
    }

    // -------------------------------
    // Final predictions
    // -------------------------------
    printf("\nFinal predictions:\n");
    for (int i = 0; i < n_samples; ++i)
    {
        printf("x = %.1f -> ŷ = %.4f (target %.1f)\n",
               mat_get(&X, i, 0),
               mat_get(&y_pred, i, 0),
               mat_get(&y, i, 0));
    }

    // -------------------------------
    // Cleanup
    // -------------------------------
    lr_free(&lr);
    mat_free(&X);
    mat_free(&y);
    mat_free(&y_pred);

    return 0;
}
