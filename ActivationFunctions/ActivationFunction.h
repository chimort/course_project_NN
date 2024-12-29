#pragma once

#include <Config.h>

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

class ReLU : public ActivationFunction
{
public:
    double evaluate(const double& x) const override;
    double derEvaluate(const double& x) const override;

    Matrix evaluate(const Matrix& x) const override;
    Matrix derEvaluate(const Matrix& x) const override;
};

class Sigmoid : public ActivationFunction
{
public:
    double evaluate(const double& x) const override;
    double derEvaluate(const double& x) const override;

    Matrix evaluate(const Matrix& x) const override;
    Matrix derEvaluate(const Matrix& x) const override;
};

class Tanh : public ActivationFunction
{
public:
    double evaluate(const double& x) const override;
    double derEvaluate(const double& x) const override;

    Matrix evaluate(const Matrix& x) const override;
    Matrix derEvaluate(const Matrix& x) const override;
};

}  // namespace neural_network
