#include "ReLU.h"

namespace neural_network
{
double ReLU::evaluate(const double& x) const { return x > 0 ? x : 0; }
double ReLU::derEvaluate(const double& x) const { return x > 0 ? 1 : 0; }

Matrix ReLU::evaluate(const Matrix& x) const { return x.cwiseMax(0); }
Matrix ReLU::derEvaluate(const Matrix& x) const { return (x.array() > 0).cast<double>(); }

}  // namespace neural_network