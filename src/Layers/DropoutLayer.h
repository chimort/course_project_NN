#pragma once

#include <EigenRand/EigenRand>
#include <random>

#include "DimensionStruct.h"
#include "Math.h"

namespace neural_network
{
class DropoutLayer
{
public:
    DropoutLayer(In in_size, Out out_size, double rate);

    Matrix evaluate(const Matrix& input) const;

    Matrix getBackpropError(const Matrix& a, const Matrix& z, const Matrix& b) const;
    void updateW(const Matrix& grad_diff, Matrix& memory, int time_step);
    void updateB(const Matrix& grad_diff, Matrix& memory, int time_step);

    Index getInputSize() const;
    Index getOutputSize() const;

private:
    double rate_;
    Index input_size_;
    Index output_size_;
    mutable Matrix last_mask_;

};

}  // namespace neural_network
