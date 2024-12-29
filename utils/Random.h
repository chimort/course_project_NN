#pragma once

#include <random>

#include "Config.h"

namespace neural_network
{
class Random
{
public:
    Random();
    Matrix generateRandomMatrix(Index rows, Index cols);

private:
    std::mt19937 generator_;
};

}  // namespace neural_network
