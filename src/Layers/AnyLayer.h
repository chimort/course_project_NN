#pragma once

#include <variant>

#include "DenseLayer.h"
#include "DropoutLayer.h"

namespace neural_network
{
class AnyLayer
{
public:
    using LayerType = std::variant<DenseLayer, DropoutLayer>;

    static AnyLayer createDenseLayer(In in_size, Out out_size, ActivationFunction f, Optimizer opt);
    static AnyLayer createDropoutLayer(In in_size, Out out_size, double rate);

    Matrix evaluate(const Matrix& input) const;
    Matrix getGradW(const Matrix& a, const Matrix& z, const Matrix& b) const;
    Matrix getGradB(const Matrix& a, const Matrix& z, const Matrix& b) const;
    Matrix getBackpropError(const Matrix& a, const Matrix& z, const Matrix& b) const;

    bool hasWeights() const;

    void updateW(const Matrix& grad_diff, Matrix& memory, int time_step);
    void updateB(const Vector& grad_diff, Vector& memory, int time_step);

    Index getInputSize() const;
    Index getOutputSize() const;

    const Matrix& getWeights() const;
    const Vector& getBiases() const;

private:
    explicit AnyLayer(LayerType layer);

    LayerType layer_;
};

} // namespace neural_network
