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
                const int n = x.cols();
                double loss = 0.0;
                for (int i = 0; i < n; ++i) {
                    const auto x_col = x.col(i);
                    const double max_coeff = x_col.maxCoeff();
                    const Eigen::ArrayXd shifted = x_col.array() - max_coeff;
                    const double sum_exp = shifted.exp().sum();
                    const double log_sum_exp = std::log(sum_exp) + max_coeff;
                    loss += -(y.col(i).array() * (x_col.array() - log_sum_exp)).sum();
                }
                return loss / n;
            },
            [](const Matrix& x, const Matrix& y) -> Matrix {
                assert(x.cols() > 0 &&
                       "Number of rows must be greater than zero to avoid division by zero");
                const int n = x.cols();
                Matrix grad(x.rows(), x.cols());
                for (int i = 0; i < n; ++i) {
                    const auto x_col = x.col(i);
                    const double max_coeff = x_col.maxCoeff();
                    const Eigen::ArrayXd shifted = x_col.array() - max_coeff;
                    const Eigen::ArrayXd exp_shifted = shifted.exp();
                    const double sum_exp = exp_shifted.sum();
                    const Eigen::ArrayXd probs = exp_shifted / sum_exp;
                    grad.col(i) = ((probs - y.col(i).array()) / static_cast<double>(n)).matrix();
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
