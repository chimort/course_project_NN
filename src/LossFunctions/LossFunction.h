#pragma once

#include <Eigen/Dense>

#include "Config.h"

namespace neural_network
{
class LossFunction
{
public:
    virtual double dist(const Matrix& x, const Matrix& y) const = 0;
    virtual Matrix derDist(const Matrix& x, const Matrix& y) const = 0;
    virtual ~LossFunction() = default;
};

}  // namespace neural_network
