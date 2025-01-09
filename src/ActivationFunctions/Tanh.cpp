#include "Tanh.h"

namespace neural_network
{
double Tanh::evaluate(const double& x) const { return tanh(x); }
double Tanh::derEvaluate(const double& x) const { return 1 - tanh(x) * tanh(x); }

Matrix Tanh::evaluate(const Matrix& x) const { return x.array().tanh().matrix(); }
Matrix Tanh::derEvaluate(const Matrix& x) const
{
    Matrix tanh_val = x.array().tanh();
    return (1.0 - tanh_val.array().square()).matrix();
}

}  // namespace neural_network