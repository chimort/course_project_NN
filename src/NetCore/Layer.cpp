#include "Layer.h"

namespace neural_network
{
// Layer
DenseLayer::DenseLayer(int input_size, int output_size, std::unique_ptr<ActivationFunction> f)
{
    assert(input_size > 0 && output_size > 0 && "Layer dimensions must be positive");
    assert(f != nullptr && "Activation function pointer is null");
    f_ = std::move(f);
    weights_ = Matrix::Random(output_size, input_size);
    biases_ = Vector::Random(output_size);
}

Matrix DenseLayer::evaluate(const Matrix& input) const
{
    assert(input.rows() == getInputSize() && "Input size mismatch");
    Matrix linear_output = (weights_ * input).colwise() + biases_;
    return f_->evaluate(linear_output);
}

Matrix DenseLayer::getGradW(const Matrix& a, const Matrix& b) const
{
    Matrix Z = weights_ * b + biases_.replicate(1, b.cols());
    Matrix activation_der = f_->derEvaluate(Z);

    assert(activation_der.rows() == a.rows() && activation_der.cols() == a.cols() &&
           "Size mismatch in getGradW");

    Matrix gradW = activation_der.cwiseProduct(a) * b.transpose();
    return gradW / b.cols();
}

Matrix DenseLayer::getGradB(const Matrix& a, const Matrix& b) const
{
    Matrix Z = weights_ * b + biases_.replicate(1, b.cols());
    Matrix activation_der = f_->derEvaluate(Z);

    assert(activation_der.rows() == a.rows() && activation_der.cols() == a.cols() &&
           "Size mismatch in getGradB");

    Matrix gradB = activation_der.cwiseProduct(a).rowwise().sum();
    return gradB / b.cols();
}

Matrix DenseLayer::getBackpropError(const Matrix& a, const Matrix& b) const
{
    Matrix Z = weights_ * b + biases_.replicate(1, b.cols());
    Matrix activation_der = f_->derEvaluate(Z);

    assert(activation_der.rows() == a.rows() && activation_der.cols() == a.cols() &&
           "Size mismatch in getNextU");

    return weights_.transpose() * (activation_der.cwiseProduct(a));
}

void DenseLayer::updateW(const Matrix& grad_diff) { weights_ -= grad_diff; }
void DenseLayer::updateB(const Matrix& grad_diff) { biases_ -= grad_diff; }

Index DenseLayer::getInputSize() const { return weights_.cols(); }
Index DenseLayer::getOutputSize() const { return weights_.rows(); }

// DropoutLayer
DropoutLayer::DropoutLayer(double rate)
    : rate_(rate), is_training_(true), input_size_(0), output_size_(0)
{
    assert(rate >= 0 && rate <= 1 && "Dropout rate must be between 0 and 1");
}

Matrix DropoutLayer::evaluate(const Matrix& input) const
{
    if (input_size_ == 0 || output_size_ == 0) {
        input_size_ = input.rows();
        output_size_ = input.cols();
    }

    if (!is_training_ || rate_ == 0) {
        return input;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(1.0 - rate_);

    Matrix mask = Matrix::NullaryExpr(input.rows(), input.cols(),
                                      [&](int, int) { return dist(gen) ? 1.0 : 0.0; });

    Matrix output = input.cwiseProduct(mask);
    return output / (1.0 - rate_);
}

Matrix DropoutLayer::getGradW(const Matrix& a, const Matrix& b) const
{
    return Matrix::Zero(a.rows(), b.cols());
}

Matrix DropoutLayer::getGradB(const Matrix& a, const Matrix& b) const
{
    return Matrix::Zero(a.rows(), b.cols());
}

Matrix DropoutLayer::getBackpropError(const Matrix& a, const Matrix& b) const
{
    return Matrix::Zero(a.rows(), b.cols());
}

void DropoutLayer::updateW(const Matrix& grad_diff) { return; }
void DropoutLayer::updateB(const Matrix& grad_diff) { return; }

Index DropoutLayer::getInputSize() const { return input_size_; }
Index DropoutLayer::getOutputSize() const { return output_size_; }

}  // namespace neural_network
