#pragma once

#include <EigenRand/EigenRand>
#include <random>

#include "Math.h"

namespace neural_network
{
class Random
{
    static constexpr Index k_default_seed = 42241;

public:
    inline Matrix generateNormalMatrix(Index rows, Index cols)
    {
        return Eigen::Rand::normal<Matrix>(rows, cols, generator_);
    }

    static Matrix generateBeoulliMatrix(Index rows, Index cols, double succes_pob)
    {
        std::bernoulli_distribution dist(succes_pob);
        return Matrix::NullaryExpr(rows, cols, [&]() { return dist(generator_) ? 1.0 : 0.0; });
    }

private:
    inline static std::mt19937 generator_ = std::mt19937{k_default_seed};
};

}  // namespace neural_network
