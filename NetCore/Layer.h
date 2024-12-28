#pragma once

#include <Eigen/Dense>
#include <random>

#include "ActivationFunction.h"
#include "Config.h"

namespace neural_network
{
class Layer
{
   public:
    Layer(int input_size, int output_size, ActivationFunction f);

   private:
    ActivationFunction f_;
    Matrix weights_;
    Vector biases_;
};

}  // namespace neural_network