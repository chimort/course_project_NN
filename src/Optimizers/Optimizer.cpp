#include "Optimizer.h"

#include <memory>

namespace neural_network
{
Optimizer::Optimizer(Signature optimizer) : optimizer_(std::move(optimizer)){}

Optimizer Optimizer::SGD(double learning_rate)
{
   return Optimizer([learning_rate](const Matrix& grad, Matrix* weights) {
        *weights -= learning_rate * grad;
    });
}

Optimizer Optimizer::Momentum(double learning_rate, double momentum) 
{
    // TODO: Implement Momentum optimizer
}

Optimizer Optimizer::Adam(double learning_rate, double beta1, double beta2, double epsilon = 1e-8) {}

void Optimizer::updateWeights(const Matrix& grad, Matrix* weights)
{
    assert(optimizer_);
    optimizer_(grad, weights);
}

} // namespace neural_network
