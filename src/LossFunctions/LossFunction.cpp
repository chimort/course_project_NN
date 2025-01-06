#include "LossFunction.h"

namespace neural_network
{
double EuclidDist::dist(const Matrix& x, const Matrix& y) const
{
    Matrix diff = x - y;
    return (diff.array().square()).sum();
}

Matrix EuclidDist::derDist(const Matrix& x, const Matrix& y) const { return 2 * (x - y); }

}  // namespace neural_network