#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <memory>
#include <random>

#include "ActivationFunction.h"
#include "Config.h"

namespace neural_network
{
class Layer
{
public:
    virtual ~Layer() = default;

    virtual Matrix evaluate(const Matrix& input) const = 0;
    virtual Matrix getGradW(const Matrix& a, const Matrix& b) const = 0;
    virtual Matrix getGradB(const Matrix& a, const Matrix& b) const = 0;
    virtual Matrix getBackpropError(const Matrix& a, const Matrix& b) const = 0;

    virtual void updateW(const Matrix& grad_diff) = 0;
    virtual void updateB(const Matrix& grad_diff) = 0;

    virtual Index getInputSize() const = 0;
    virtual Index getOutputSize() const = 0;
};

}  // namespace neural_network