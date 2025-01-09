#pragma once

#include "Config.h"

namespace neural_network
{
class ActivationFunction
{
public:
    virtual double evaluate(const double& x) const = 0;
    virtual double derEvaluate(const double& x) const = 0;

    virtual Matrix evaluate(const Matrix& x) const = 0;
    virtual Matrix derEvaluate(const Matrix& x) const = 0;

    virtual ~ActivationFunction() = default;
};

}  // namespace neural_network
