#pragma once

#include<vector>
#include<iostream>
#include<cmath>

using VectorDouble = std::vector<double>;

class RosenbrockCost
{
public:
    RosenbrockCost() = default;
    ~RosenbrockCost() = default;
    double ComputeValue(const VectorDouble &x);
    VectorDouble ComputeGradient(const VectorDouble &x);
    double ComputeGradientNorm(const VectorDouble &x);
};

