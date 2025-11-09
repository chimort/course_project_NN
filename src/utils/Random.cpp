#include <Random.h>

namespace neural_network
{
thread_local std::mt19937 Random::generator_{Random::k_default_seed};

Matrix Random::generateNormalMatrix(Index rows, Index cols)
{
    return Eigen::Rand::normal<Matrix>(rows, cols, generator_);
}

Matrix Random::generateBernoulliMatrix(Index rows, Index cols, double success_prob)
{
    std::bernoulli_distribution dist(success_prob);
    return Matrix::NullaryExpr(rows, cols, [&]() { return dist(generator_) ? 1.0 : 0.0; });
}

}  // namespace neural_network
