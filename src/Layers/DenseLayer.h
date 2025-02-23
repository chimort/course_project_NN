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
    Matrix getGradW(const Matrix& a, const Matrix& z, const Matrix& b) const;
    Matrix getGradB(const Matrix& a, const Matrix& z, const Matrix& b) const;
    Matrix getBackpropError(const Matrix& a, const Matrix& z, const Matrix& b) const;

    void updateW(const Matrix& grad_diff, Matrix& memory, int time_step);
    void updateB(const Vector& grad_diff, Vector& memory, int time_step);

    Index getInputSize() const;
    Index getOutputSize() const;

    inline const Matrix& getWeights() const { return weights_; }
    inline const Vector& getBiases() const { return biases_; }

    Matrix applyActivation(const Matrix& z) const;

private:
    Random rnd_;
    ActivationFunction f_;
    Matrix weights_;
    Vector biases_;
    Optimizer opt_;
};

}  // namespace neural_network
