#pragma once

#include "Optimizer.h"

namespace neural_network
{
class Adam : public Optimizer
{
public:
    Adam(double learnign_rate, double beta1, double beta2, double eps);

    void updateWeights(Matrix& w, const Matrix& grad_w) override;

private:
    double learning_rate_;
    double beta1_;
    double beta2_;
    double eps_;
    Matrix m_;
    Matrix v_;
};
} // namespace neural_network