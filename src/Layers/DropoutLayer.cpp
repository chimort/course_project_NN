#include "DropoutLayer.h"

namespace neural_network
{
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
    
} // namespace neural_network

