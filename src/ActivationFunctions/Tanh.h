#pragma once

#include "ActivationFunction.h"

namespace neural_network
{
class Tanh : public ActivationFunction
{
public:
    double evaluate(const double& x) const override;
    double derEvaluate(const double& x) const override;

    Matrix evaluate(const Matrix& x) const override;
    Matrix derEvaluate(const Matrix& x) const override;
};

}  // namespace neural_network