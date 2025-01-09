#pragma once

#include "Layer.h"

namespace neural_network
{
class DenseLayer : public Layer
{
public:
    DenseLayer(int input_size, int output_size, std::unique_ptr<ActivationFunction> f);

    Matrix evaluate(const Matrix& input) const override;
    Matrix getGradW(const Matrix& a, const Matrix& b) const override;
    Matrix getGradB(const Matrix& a, const Matrix& b) const override;
    Matrix getBackpropError(const Matrix& a, const Matrix& b) const override;

    void updateW(const Matrix& grad_diff) override;
    void updateB(const Matrix& grad_diff) override;

    Index getInputSize() const override;
    Index getOutputSize() const override;

private:
    std::unique_ptr<ActivationFunction> f_;
    Matrix weights_;
    Vector biases_;
};

} // namespace neural_network