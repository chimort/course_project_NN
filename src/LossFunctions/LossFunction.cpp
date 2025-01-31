#include "LossFunction.h"

namespace neural_network 
{
LossFunction LossFunction::Euclid() 
{
    return {
        [](const Matrix& x, const Matrix& y) 
        { 
            assert(x.rows() == y.rows() && x.cols() == y.cols() && 
                "Input matrices must have the same dimensions");
            return (x - y).squaredNorm() / x.size(); 
        },
        [](const Matrix& x, const Matrix& y) { return 2 * (x - y) / x.size(); }
    };
}

double LossFunction::dist(const Matrix& x, const Matrix& y) const
{
    assert(f0_);
    return f0_(x, y);
}

Matrix LossFunction::derDist(const Matrix& x, const Matrix& y) const
{
    assert(f1_);
    return f1_(x, y);
}

} // neural_network