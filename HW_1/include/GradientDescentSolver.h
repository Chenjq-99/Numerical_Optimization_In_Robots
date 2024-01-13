#pragma once

#include<vector>
#include<iostream> 
#include "../include/RosenbrockCost.h"
#include "../include/Logger.h"

using VectorDouble = std::vector<double>;

class GradientDescentSolver
{
public:
    GradientDescentSolver(const VectorDouble& init_guess);
    GradientDescentSolver(const VectorDouble& init_guess, const int& maxIters, const double& c);
    ~GradientDescentSolver() = default;
    VectorDouble Solve();
    double LineSearch(const VectorDouble &x);
    VectorDouble GetNextXWithStepAndDirection(const VectorDouble &x, 
                            const VectorDouble &direction, const double alpha);
private:
    RosenbrockCost cost_;
    VectorDouble init_guess_;
    Logger logger_;
    int max_iters_ = 1000;
    double c_ = 1e-4;
    double discount_ = 0.5;
    double alpha_ = 1.0;
};
