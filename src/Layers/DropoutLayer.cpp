#include "DropoutLayer.h"

namespace neural_network
{
DropoutLayer::DropoutLayer(In in_size, Out out_size, double rate)
    : input_size_(in_size.value), output_size_(out_size.value), rate_(rate)
{
    assert(rate >= 0 && rate <= 1 && "Dropout rate must be between 0 and 1");
}

Matrix DropoutLayer::evaluate(const Matrix& input) const
{
    if (rate_ == 0) {
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

Matrix DropoutLayer::getBackpropError(const Matrix& a, const Matrix& z, const Matrix& b) const
{
    return a;
}

void DropoutLayer::updateW(const Matrix& grad_diff, Matrix& memory, int time_step) { return; }
void DropoutLayer::updateB(const Matrix& grad_diff, Matrix& memory, int time_step) { return; }

Index DropoutLayer::getInputSize() const { return input_size_; }
Index DropoutLayer::getOutputSize() const { return output_size_; }

}  // namespace neural_network
