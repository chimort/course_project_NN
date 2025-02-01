#pragma once

#include "ActivationFunction.h"
#include "Math.h"
#include "Random.h"

namespace neural_network
{
class DenseLayer
{
public:
    struct In {
        Index value;
        explicit constexpr In(Index v) : value(v) {}
    };

    struct Out {
        Index value;
        explicit constexpr Out(Index v) : value(v) {}
    };

    DenseLayer(In in_size, Out out_size, ActivationFunction f);

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
};

}  // namespace neural_network
