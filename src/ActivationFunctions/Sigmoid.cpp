#include "Sigmoid.h"

namespace neural_network
{
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

}  // namespace neural_network