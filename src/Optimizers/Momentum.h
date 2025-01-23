#pragma once

#include "Optimizer.h"

namespace neural_network
{
class Momentum : public Optimizer
{
public:
    Momentum(double learning_rate, double momentum)
        : learning_rate_(learning_rate), momentum_(momentum)
    {
    }

    inline void updateWeights(Matrix& w, const Matrix& grad_w) override
    {
        if (velocity_.size() == 0) {
            velocity_ = Matrix::Zero(w.rows(), w.cols());
        }

        velocity_ = momentum_ * velocity_ - learning_rate_ * grad_w;
        w += velocity_;
    }

private:
    double learning_rate_;
    double momentum_;
    Matrix velocity_;
};
}  // namespace neural_network
