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

LossFunction LossFunction::CrossEntropy()
{
    constexpr float epsilon = 1e-8f;
    return {[epsilon](const Matrix& x, const Matrix& y) -> double {
                assert(x.cols() > 0 &&
                       "Number of rows must be greater than zero to avoid division by zero");
                return -(y.array() * (x.array() + epsilon).log()).sum() / x.cols();
            },
            [epsilon](const Matrix& x, const Matrix& y) -> Matrix {
                assert(x.cols() > 0 &&
                       "Number of rows must be greater than zero to avoid division by zero");
                return -(y.array() / (x.array() + epsilon)).matrix() / x.cols();
            }};
}

LossFunction LossFunction::CrossEntropyWithLogits()
{
    return {[](const Matrix& x, const Matrix& y) -> double {
                assert(x.cols() > 0 &&
                       "Number of rows must be greater than zero to avoid division by zero");
                double totalLoss = 0.0;
                int n = x.cols();
                for (int j = 0; j < n; j++) {
                    auto col_x = x.col(j);
                    double maxCoeff = col_x.maxCoeff();
                    Eigen::ArrayXd shifted = col_x.array() - maxCoeff;
                    double sumExp = shifted.exp().sum();
                    double logSumExp = std::log(sumExp) + maxCoeff;
                    totalLoss += -(y.col(j).array() * (col_x.array() - logSumExp)).sum();
                }
                return totalLoss / n;
            },
            [](const Matrix& x, const Matrix& y) -> Matrix {
                assert(x.cols() > 0 &&
                       "Number of rows must be greater than zero to avoid division by zero");
                Matrix grad(x.rows(), x.cols());
                int n = x.cols();
                for (int j = 0; j < n; j++) {
                    auto col_x = x.col(j);
                    double maxCoeff = col_x.maxCoeff();
                    Eigen::ArrayXd shifted = col_x.array() - maxCoeff;
                    double sumExp = shifted.exp().sum();
                    Eigen::ArrayXd softmax = shifted.exp() / sumExp;
                    grad.col(j) = ((softmax - y.col(j).array()) / static_cast<double>(n)).matrix();
                }
                return grad;
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
