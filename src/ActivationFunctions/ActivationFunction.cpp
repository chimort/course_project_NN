#include "ActivationFunction.h"

namespace neural_network
{
// ReLU
double ReLU::evaluate(const double& x) const { return x > 0 ? x : 0; }
double ReLU::derEvaluate(const double& x) const { return x > 0 ? 1 : 0; }

Matrix ReLU::evaluate(const Matrix& x) const { return x.cwiseMax(0); }
Matrix ReLU::derEvaluate(const Matrix& x) const { return (x.array() > 0).cast<double>(); }

// Sigmoid
double Sigmoid::evaluate(const double& x) const { return 1 / (1 + exp(-x)); }
double Sigmoid::derEvaluate(const double& x) const
{
    double sigmoid = Sigmoid::evaluate(x);
    return sigmoid * (1.0 - sigmoid);
}

Matrix Sigmoid::evaluate(const Matrix& x) const
{
    return (1.0 / (1.0 + (-x.array()).exp())).matrix();
}
Matrix Sigmoid::derEvaluate(const Matrix& x) const
{
    Matrix sigmoid = Sigmoid::evaluate(x);
    return (sigmoid.array() * (1.0 - sigmoid.array())).matrix();
}

// Tanh
double Tanh::evaluate(const double& x) const { return tanh(x); }
double Tanh::derEvaluate(const double& x) const { return 1 - tanh(x) * tanh(x); }

Matrix Tanh::evaluate(const Matrix& x) const { return x.array().tanh().matrix(); }
Matrix Tanh::derEvaluate(const Matrix& x) const
{
    Matrix tanh_val = x.array().tanh();
    return (1.0 - tanh_val.array().square()).matrix();
}

}  // namespace neural_network