#include "Optimizer.h"

#include <memory>

namespace neural_network
{
Optimizer::Optimizer(Signature optimizer) : optimizer_(std::move(optimizer)) {}

Optimizer Optimizer::SGD(double learning_rate)
{
    return Optimizer(
        [learning_rate](const Matrix& grad, Matrix* weights) { *weights -= learning_rate * grad; });
}

Optimizer Optimizer::Momentum(double learning_rate, double beta1)
{
    auto m = std::make_shared<Matrix>();

    return Optimizer([m, learning_rate, beta1](const Matrix& grad, Matrix* weights) {
        if (m->rows() != grad.rows() || m->cols() != grad.cols()) {
            *m = Matrix::Zero(grad.rows(), grad.cols());
        }

        *m = beta1 * (*m) + (1 - beta1) * grad;

        *weights -= learning_rate * (*m);
    });
}

Optimizer Optimizer::Adam(double learning_rate, double beta1, double beta2, double epsilon)
{
    auto m = std::make_shared<Matrix>();
    auto v = std::make_shared<Matrix>();
    auto t = std::make_shared<int>(0);

    return Optimizer([m, v, t, learning_rate, beta1, beta2, epsilon](const Matrix& grad,
                                                                     Matrix* weights) mutable {
        if (m->rows() != grad.rows() || m->cols() != grad.cols()) {
            *m = Matrix::Zero(grad.rows(), grad.cols());
            *v = Matrix::Zero(grad.rows(), grad.cols());
        }

        ++(*t);
        *m = beta1 * (*m) + (1 - beta1) * grad;
        *v = beta2 * (*v) + (1 - beta2) * grad.array().square().matrix();
        Matrix mHat = (*m) / (1 - std::pow(beta1, *t));
        Matrix vHat = (*v) / (1 - std::pow(beta2, *t));
        *weights -= (learning_rate * mHat.array() / (vHat.array().sqrt() + epsilon)).matrix();
    });
}

void Optimizer::updateWeights(const Matrix& grad, Matrix* weights)
{
    assert(optimizer_);
    optimizer_(grad, weights);
}

}  // namespace neural_network
