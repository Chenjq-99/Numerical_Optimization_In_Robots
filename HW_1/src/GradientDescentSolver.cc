#include "../include/GradientDescentSolver.h"

GradientDescentSolver::GradientDescentSolver(const VectorDouble& init_guess) 
                        : init_guess_(init_guess) {}
GradientDescentSolver::GradientDescentSolver(const VectorDouble& init_guess, const int& max_iters, const double& c)
                        : init_guess_(init_guess), max_iters_(max_iters), c_(c) {}

VectorDouble GradientDescentSolver::Solve() {
    double kEPSILON = 1e-5;
    auto x = init_guess_;
    int iter_count = 0;
    double delta = cost_.ComputeGradientNorm(x);
    while (delta >= kEPSILON && iter_count <= max_iters_) {
        logger_.LOG(iter_count, x, delta);
        double alpha = LineSearch(x);
        auto gradient = cost_.ComputeGradient(x);
        x = GetNextXWithStepAndDirection(x, gradient, alpha);
        delta = cost_.ComputeGradientNorm(x);
        iter_count++;
    }
    return x;
}

double GradientDescentSolver::LineSearch(const VectorDouble &x) {
    double item_value = 0.0;
    double alpha = alpha_;
    VectorDouble gradient = cost_.ComputeGradient(x);
    int gradient_size = gradient.size();
    VectorDouble direction = gradient;
    VectorDouble x_new = GetNextXWithStepAndDirection(x, direction, alpha);
    
    for (int i = 0; i < gradient_size; ++i) {
        item_value += alpha * c_ * direction[i] * gradient[i];
    }
    while (cost_.ComputeValue(x_new) >= cost_.ComputeValue(x) - item_value) {
        alpha = discount_ * alpha;
        x_new = GetNextXWithStepAndDirection(x, direction, alpha);
        item_value = 0.0;
        for (int i = 0; i < gradient_size; ++i) {
            item_value += alpha * c_ * direction[i] * gradient[i];
        }
    }
    return alpha;
}

VectorDouble GradientDescentSolver::GetNextXWithStepAndDirection(const VectorDouble &x, 
                            const VectorDouble &direction, const double alpha) {
    int x_size = x.size();
    VectorDouble x_new(x_size);    
    for (int i = 0; i < x_size; ++i) {
        x_new[i] = x[i] - alpha * direction[i];
    }
    return x_new;                            
}