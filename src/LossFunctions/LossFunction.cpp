#include "LossFunction.h"

namespace neural_network
{
double EuclidDist::dist(const Matrix& x, const Matrix& y) const
{
    assert(x.rows() == y.rows() && x.cols() == y.cols() &&
           "Input matrices must have the same dimensions");

    Matrix diff = x - y;
    return diff.squaredNorm();
}

Matrix EuclidDist::derDist(const Matrix& x, const Matrix& y) const { return 2 * (x - y); }

}  // namespace neural_network