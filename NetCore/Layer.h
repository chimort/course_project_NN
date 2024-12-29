#pragma once

#include <Eigen/Dense>
#include <random>

#include "ActivationFunction.h"
#include "Config.h"

namespace neural_network
{
class Layer
{
public:
    Layer(int input_size, int output_size, ActivationFunction f);
    Matrix evaluate(const Matrix& input) const;
    Matrix getGradW(const Matrix& a, const Matrix& b) const;
    Matrix getGradB(const Matrix& a, const Matrix& b) const;
    Matrix getNextU(const Matrix& a, const Matrix& b) const;
    void updateW(const Matrix& grad_diff);
    void updateB(const Matrix& grad_diff);
    Index getInputSize() const;
    Index getOutputSize() const;

private:
    ActivationFunction f_;
    Matrix weights_;
    Vector biases_;
};

}  // namespace neural_network