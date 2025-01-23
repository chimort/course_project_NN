#include "DenseLayer.h"

namespace neural_network
{
DenseLayer::DenseLayer(int input_size, int output_size, std::unique_ptr<ActivationFunction> f)
{
    assert(input_size > 0 && output_size > 0 && "Layer dimensions must be positive");
    assert(f != nullptr && "Activation function pointer is null");
    f_ = std::move(f);
    weights_ = Matrix::Random(output_size, input_size);
    biases_ = Vector::Random(output_size);
}

Matrix DenseLayer::evaluate(const Matrix& input) const
{
    assert(input.cols() == getInputSize() && "Input size mismatch");

    Matrix linear_output = (weights_ * input.transpose()).colwise() + biases_;
    return f_->evaluate(linear_output.transpose());
}

Matrix DenseLayer::getGradW(const Matrix& grad, const Matrix& input_data) const
{
    Matrix gradW = Matrix::Zero(weights_.rows(), weights_.cols());

    for (Index i = 0; i < input_data.rows(); ++i) {
        Matrix z = weights_ * input_data.row(i).transpose() + biases_;
        gradW += (f_->derEvaluate(z).asDiagonal() * grad.row(i).transpose() * input_data.row(i));
    }

    return gradW / input_data.rows();
}

Matrix DenseLayer::getGradB(const Matrix& grad, const Matrix& input_data) const
{
    Matrix gradB = Vector::Zero(biases_.size());

    for (Index i = 0; i < input_data.rows(); ++i) {
        Matrix z = weights_ * input_data.row(i).transpose() + biases_;
        gradB += (f_->derEvaluate(z).asDiagonal() * grad.row(i).transpose());
    }

    return gradB / input_data.rows();
}

Matrix DenseLayer::getBackpropError(const Matrix& grad, const Matrix& input_data) const
{
    assert(input_data.cols() == weights_.cols() && "Input dimensions mismatch in getBackpropError");

    Matrix error = Matrix::Zero(grad.rows(), weights_.cols());

    for (Index i = 0; i < input_data.rows(); ++i) {
        Matrix z = weights_ * input_data.row(i).transpose() + biases_;
        error.row(i) = (weights_.transpose() * f_->derEvaluate(z).asDiagonal() *
                        grad.row(i).transpose())
                           .transpose();
    }

    return error;
}

void DenseLayer::updateW(const Matrix& grad_diff) { weights_ -= grad_diff; }
void DenseLayer::updateB(const Matrix& grad_diff) { biases_ -= grad_diff; }

Index DenseLayer::getInputSize() const { return weights_.cols(); }
Index DenseLayer::getOutputSize() const { return weights_.rows(); }

}  // namespace neural_network