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

}  // namespace neural_network