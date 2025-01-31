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

private:
    std::mt19937 generator_ = std::mt19937{k_default_seed};
};

}  // namespace neural_network
