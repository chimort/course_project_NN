#pragma once

#include "ActivationFunction.h"
#include "Math.h"
#include "Optimizer.h"
#include "Random.h"

namespace neural_network
{
class DenseLayer
{
public:
    struct In {
        Index value;
    };

    struct Out {
        Index value;
    };

    DenseLayer(In in_size, Out out_size, ActivationFunction f, Optimizer opt);

    Matrix evaluate(const Matrix& input) const;
    Matrix getGradW(const Matrix& a, const Matrix& b) const;
    Matrix getGradB(const Matrix& a, const Matrix& b) const;
    Matrix getBackpropError(const Matrix& a, const Matrix& b) const;

    void updateW(const Matrix& grad_diff);
    void updateB(const Matrix& grad_diff);

    Index getInputSize() const;
    Index getOutputSize() const;

private:
    Random rnd_;
    ActivationFunction f_;
    Matrix weights_;
    Vector biases_;
    Optimizer opt_;
};

}  // namespace neural_network
