#pragma once

#include "Layer.h"

namespace neural_network
{
class DropoutLayer : public Layer
{
public:
    DropoutLayer(double rate);

    inline void setTrainingMode(bool is_training) const { is_training_ = is_training; };

    Matrix evaluate(const Matrix& input) const override;
    Matrix getGradW(const Matrix& a, const Matrix& b) const override;
    Matrix getGradB(const Matrix& a, const Matrix& b) const override;
    Matrix getBackpropError(const Matrix& a, const Matrix& b) const override;
    void updateW(const Matrix& grad_diff) override;
    void updateB(const Matrix& grad_diff) override;

    Index getInputSize() const override;
    Index getOutputSize() const override;

private:
    mutable bool is_training_;
    double rate_;
    mutable Index input_size_;
    mutable Index output_size_;
};
} // namespace neural_network