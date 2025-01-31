#include "ActivationFunction.h"

namespace neural_network
{

ActivationFunction::ActivationFunction(Function f0, Function f1)
    : f0_(std::move(f0)), f1_(std::move(f1))
{
}

ActivationFunction ActivationFunction::ReLU()
{
    return ActivationFunction([](double x) { return (x > 0) * x; },
                              [](double x) { return (x > 0); });
}

ActivationFunction ActivationFunction::Sigmoid()
{
    return ActivationFunction([](double x) { return 1 / (1 + exp(-x)); },
                              [](double x) {
                                  double s = 1 / (1 + exp(-x));
                                  return s * (1 - s);
                              });
}

double ActivationFunction::evaluate(double x) const
{
    assert(f0_);
    return f0_(x);
}

double ActivationFunction::derEvaluate(double x) const
{
    assert(f1_);
    return f1_(x);
}

Matrix ActivationFunction::evaluate(const Matrix& x) const
{
    assert(f0_);
    return x.unaryExpr(f0_);
}

Matrix ActivationFunction::derEvaluate(const Matrix& x) const
{
    assert(f1_);
    return x.unaryExpr(f1_);
}

}  // namespace neural_network
