#include "DropoutLayer.h"

#include <iostream>

namespace neural_network
{
DropoutLayer::DropoutLayer(In in_size, Out out_size, double rate)
    : input_size_(in_size), output_size_(out_size), rate_(rate)
{
    assert(rate >= 0 && rate <= 1 && "Dropout rate must be between 0 and 1");
}

Matrix DropoutLayer::evaluate(const Matrix& input) const { return input; }

Matrix DropoutLayer::evaluate(const Matrix& input, DropoutCache& cache) const
{
    if (rate_ == 0.0) {
        return input;
    }

    cache.mask = Random::generateBernoulliMatrix(input.rows(), input.cols(), 1.0 - rate_);

    return input.cwiseProduct(cache.mask) / (1.0 - rate_);
}

Matrix DropoutLayer::getBackpropError(const Matrix& a, const Matrix& z, const Matrix& x,
                                      const DropoutCache& cache) const
{
    if (rate_ == 0.0) {
        return a;
    }

    return a.cwiseProduct(cache.mask) / (1.0 - rate_);
}

void DropoutLayer::updateW(const Matrix& grad_diff, Matrix& memory, int time_step) { return; }
void DropoutLayer::updateB(const Matrix& grad_diff, Matrix& memory, int time_step) { return; }

Index DropoutLayer::getInputSize() const { return input_size_; }
Index DropoutLayer::getOutputSize() const { return output_size_; }

}  // namespace neural_network
