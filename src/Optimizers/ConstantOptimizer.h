#pragma once

#include "Optimizer.h"

namespace neural_network
{
class ConstantOptimizer : public Optimizer
{
public:
    ConstantOptimizer(double learning_rate) : learning_rate_(learning_rate) {};

    inline void updateWeights(Matrix& w, const Matrix& grad_w) override { w -= learning_rate_ * grad_w; }

private:
    double learning_rate_;
};
} // namespace neural_network