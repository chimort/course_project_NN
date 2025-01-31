#pragma once

#include <Eigen/Dense>

#include "Math.h"

namespace neural_network
{
class Optimizer
{
public:
    virtual ~Optimizer() = default;

    virtual void updateWeights(Matrix& w, const Matrix& grad_w) = 0;
};
}  // namespace neural_network
