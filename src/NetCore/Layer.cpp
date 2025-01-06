#include "Layer.h"

namespace neural_network
{
Layer::Layer(int input_size, int output_size, ActivationFunction* f)
{
    assert(input_size > 0 && output_size > 0 && "Layer dimensions must be positive");
    assert(f != nullptr && "Activation function pointer is null");
    f_ = f;
    weights_ = Matrix::Random(output_size, input_size);
    biases_ = Vector::Random(output_size);
}

Matrix Layer::evaluate(const Matrix& input) const
{
    assert(input.cols() == getInputSize() && "Input size mismatch");
    Matrix linear_output = (weights_ * input).colwise() + biases_;
    return f_->evaluate(linear_output);
}

Matrix Layer::getGradW(const Matrix& a, const Matrix& b) const
{
    Matrix Z = weights_ * b + biases_.replicate(1, b.cols());
    Matrix activation_der = f_->derEvaluate(Z);
    Matrix gradW = (a.transpose() * activation_der).transpose() * b.transpose();
    return gradW / b.cols();
}

Matrix Layer::getGradB(const Matrix& a, const Matrix& b) const
{
    // тоже самое, что и выше
    return a.rowwise().sum();
}

Matrix Layer::getNextU(const Matrix& a, const Matrix& b) const
{
    // тоже самое, что и выше
    return a.transpose() * b;
}

void Layer::updateW(const Matrix& grad_diff) { weights_ -= grad_diff; }

void Layer::updateB(const Matrix& grad_diff) { biases_ -= grad_diff; }

Index Layer::getInputSize() const { return weights_.cols(); }

Index Layer::getOutputSize() const { return weights_.rows(); }

}  // namespace neural_network
