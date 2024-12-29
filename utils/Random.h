#pragma once

#include <EigenRand/EigenRand>
#include <random>

#include "Config.h"

namespace neural_network
{
class Random
{
public:
    Random() : generator_(std::mt19937(42241)) {}
    inline Matrix generateRandomMatrix(Index rows, Index cols)
    {
        return Eigen::Rand::normal<Matrix>(rows, cols, generator_);
    }

private:
    std::mt19937 generator_;
};

}  // namespace neural_network
