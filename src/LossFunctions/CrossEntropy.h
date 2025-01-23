#include "LossFunction.h"

namespace neural_network
{
class CrossEntropy : public LossFunction
{
public:
    double dist(const Matrix& x, const Matrix& y) const override;
    Matrix derDist(const Matrix& x, const Matrix& y) const override;
};

} // namespace neural_network