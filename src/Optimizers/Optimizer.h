#pragma once

#include <Eigen/Dense>
#include <functional>

#include "Math.h"

namespace neural_network
{
class Optimizer
{
    using Signature = std::function<void(const Matrix&, Matrix*)>;
    Optimizer(Signature optimizer);

public:
    static Optimizer SGD(double learning_rate);
    static Optimizer Momentum(double learning_rate, double beta1);
    static Optimizer Adam(double learning_rate, double beta1, double beta2, double epsilon = 1e-8);

    void updateWeights(const Matrix& grad, Matrix* weights);
    void updateWeights(const Matrix& grad, Vector* weights);

private:
    Signature optimizer_;
};

}  // namespace neural_network
