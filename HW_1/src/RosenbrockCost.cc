#include "RosenbrockCost.h"

double RosenbrockCost::ComputeValue(const VectorDouble &x) {
    int x_size = x.size();
    // https://en.wikipedia.org/wiki/Rosenbrock_function
    double value = 0.0;
    for (int i = 0; i < x_size - 1; i++) {
        value += 100.0 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i]) + (1 - x[i]) * (1 - x[i]);
    }
    return value;
}

VectorDouble RosenbrockCost::ComputeGradient(const VectorDouble &x) {
    int x_size = x.size();
    VectorDouble gradient(x_size);
    for (int i = 0; i < x_size; ++i) {
        if (i == 0) {
            gradient[i] =  -400 * x[i] * (x[i + 1] - x[i] * x[i]) - 2 * (1 - x[i]);
        } else if (i == x_size - 1) {
            gradient[i] = 200 * (x[i] - x[i - 1] * x[i - 1]);
        } else {
            gradient[i] = -400 * x[i] * (x[i + 1] - x[i] * x[i]) - 2 * (1 - x[i]) + 200 * (x[i] - x[i - 1] * x[i - 1]);
        }
    }
    return gradient;
}

double RosenbrockCost::ComputeGradientNorm(const VectorDouble &x) {
    auto gradient = ComputeGradient(x);
    int gradient_size = gradient.size();
    double norm_square = 0.0;
    for (int i = 0; i < gradient_size; ++i) {
        norm_square += gradient[i] * gradient[i];
    }
    return std::sqrt(norm_square);
}