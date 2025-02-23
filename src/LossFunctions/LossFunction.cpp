#include "LossFunction.h"

namespace neural_network
{
LossFunction::LossFunction(Signature1 f0, Signature2 f1) : f0_(std::move(f0)), f1_(std::move(f1)) {}

LossFunction LossFunction::Euclid()
{
    return {[](const Matrix& x, const Matrix& y) {
                assert(x.rows() == y.rows() && x.cols() == y.cols() &&
                       "Input matrices must have the same dimensions");
                assert(x.cols() > 0 &&
                       "Number of rows must be greater than zero to avoid division by zero");
                return 0.5 * (x - y).squaredNorm() / x.cols();
            },
            [](const Matrix& x, const Matrix& y) {
                assert(x.cols() > 0 &&
                       "Number of rows must be greater than zero to avoid division by zero");
                return (x - y) / x.cols();
            }};
}

double LossFunction::dist(const Matrix& x, const Matrix& y) const
{
    assert(f0_);
    return f0_(x, y);
}

Matrix LossFunction::derDist(const Matrix& x, const Matrix& y) const
{
    assert(f1_);
    return f1_(x, y);
}

}  // namespace neural_network
