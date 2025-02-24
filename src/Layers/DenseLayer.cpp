#include "DenseLayer.h"

#include <cassert>

namespace neural_network
{
DenseLayer::DenseLayer(In input_size, Out output_size, ActivationFunction f, Optimizer opt)
    : rnd_(Random()),
      weights_(rnd_.generateNormalMatrix(output_size.value, input_size.value)),
      biases_(rnd_.generateNormalMatrix(output_size.value, 1)),
      f_(std::move(f)),
      opt_(std::move(opt))
{
    assert(input_size.value > 0 && output_size.value > 0 && "Layer dimensions must be positive");
}

Matrix DenseLayer::evaluate(const Matrix& input) const
{
    assert(input.rows() == getInputSize() && "Input size mismatch");
    return f_.evaluate((weights_ * input).colwise() + biases_);
}

Matrix DenseLayer::getGradW(const Matrix& grad, const Matrix& z, const Matrix& input_data) const
{
    assert(weights_.cols() == input_data.rows() && "weights_ and input_data have different sizes");
    assert(grad.cols() == input_data.cols() && "grad and input_data have different batch sizes");
    assert(biases_.rows() == weights_.rows() && "biases and weights_ have different sizes");
    assert(input_data.rows() > 0 && "input_data rows size have to be greater than 0");

    return (f_.derEvaluate(z).array() * grad.array()).matrix() * input_data.transpose() /
           input_data.cols();
}

Matrix DenseLayer::getGradB(const Matrix& grad, const Matrix& z, const Matrix& input_data) const
{
    assert(weights_.cols() == getInputSize() && "weights_ cols mismatch with input size");
    assert(weights_.rows() == getOutputSize() && "weights_ rows mismatch with output size");
    assert(grad.rows() == biases_.rows() && "grad and biases_ have different sizes");

    return (f_.derEvaluate(z).array() * grad.array()).rowwise().mean();
}

Matrix DenseLayer::getBackpropError(const Matrix& grad, const Matrix& z,
                                    const Matrix& input_data) const
{
    assert(input_data.rows() == weights_.cols() && "Input dimensions mismatch in getBackpropError");

    return weights_.transpose() * (f_.derEvaluate(z).array() * grad.array()).matrix();
}

void DenseLayer::updateW(const Matrix& grad_diff, Matrix& memory, int time_step)
{
    weights_ -= opt_.getUpdateA(grad_diff, weights_, memory, time_step);
}

void DenseLayer::updateB(const Vector& grad_diff, Vector& memory, int time_step)
{
    biases_ -= opt_.getUpdateB(grad_diff, biases_, memory, time_step);
}

Index DenseLayer::getInputSize() const { return weights_.cols(); }
Index DenseLayer::getOutputSize() const { return weights_.rows(); }

}  // namespace neural_network
