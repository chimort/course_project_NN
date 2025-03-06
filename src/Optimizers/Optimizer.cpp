#include "Optimizer.h"

#include <memory>

namespace neural_network
{
Optimizer::Optimizer(MSignature optimizerA, VSignature optimizerB)
    : optimizerA_(std::move(optimizerA)), optimizerB_(std::move(optimizerB))
{
}

Optimizer Optimizer::SGD(double learning_rate)
{
    MSignature sgdUpdateA = [learning_rate](const Matrix& grad, const Matrix& currentWeights,
                                            Matrix& memory, int time_step) -> Matrix {
        return learning_rate * grad;
    };

    VSignature sgdUpdateB = [learning_rate](const Vector& grad, const Vector& currentBiases,
                                            Vector& memory, int time_step) -> Vector {
        return learning_rate * grad;
    };

    return Optimizer(sgdUpdateA, sgdUpdateB);
}

Optimizer Optimizer::Momentum(double learning_rate, double beta1)
{
    MSignature momentum_update_a = [learning_rate,
                                    beta1](const Matrix& grad, const Matrix& currentWeights,
                                           Matrix& memory, Index time_step) -> Matrix {
        if (memory.rows() != grad.rows() || memory.cols() != grad.cols()) {
            memory = Matrix::Zero(grad.rows(), grad.cols());
        }
        memory = beta1 * memory + (1 - beta1) * grad;
        return learning_rate * memory;
    };

    VSignature momentum_update_b = [learning_rate,
                                    beta1](const Vector& grad, const Vector& currentBiases,
                                           Vector& memory, Index time_step) -> Vector {
        if (memory.rows() != grad.rows() || memory.cols() != grad.cols()) {
            memory = Matrix::Zero(grad.rows(), grad.cols());
        }
        memory = beta1 * memory + (1 - beta1) * grad;
        return learning_rate * memory;
    };

    return Optimizer(momentum_update_a, momentum_update_b);
}

Optimizer Optimizer::Adam(double learning_rate, double beta1, double beta2, double epsilon)
{
    MSignature adamUpdateA = [learning_rate, beta1, beta2,
                              epsilon](const Matrix& grad, const Matrix& currentWeights,
                                       Matrix& memory, Index time_step) -> Matrix {
        if (memory.size() == 0 || memory.rows() != 2 * grad.rows() ||
            memory.cols() != grad.cols()) {
            memory = Matrix::Zero(2 * grad.rows(), grad.cols());
        }
        Matrix m = memory.topRows(grad.rows());
        Matrix v = memory.bottomRows(grad.rows());

        m = beta1 * m + (1 - beta1) * grad;
        v = beta2 * v + (1 - beta2) * grad.array().square().matrix();

        memory.topRows(grad.rows()) = m;
        memory.bottomRows(grad.rows()) = v;

        Matrix mHat = m / (1 - std::pow(beta1, time_step));
        Matrix vHat = v / (1 - std::pow(beta2, time_step));
        return learning_rate * (mHat.array() / (vHat.array().sqrt() + epsilon)).matrix();
    };

    VSignature adamUpdateB = [learning_rate, beta1, beta2,
                              epsilon](const Vector& grad, const Vector& currentBiases,
                                       Vector& memory, Index time_step) -> Vector {
        if (memory.size() == 0 || memory.size() != 2 * grad.size()) {
            memory = Vector::Zero(2 * grad.size());
        }
        Vector m = memory.head(grad.size());
        Vector v = memory.tail(grad.size());

        m = beta1 * m + (1 - beta1) * grad;
        v = beta2 * v + (1 - beta2) * grad.array().square().matrix();

        memory.head(grad.size()) = m;
        memory.tail(grad.size()) = v;

        Vector mHat = m / (1 - std::pow(beta1, time_step));
        Vector vHat = v / (1 - std::pow(beta2, time_step));
        return learning_rate * (mHat.array() / (vHat.array().sqrt() + epsilon)).matrix();
    };

    return Optimizer(adamUpdateA, adamUpdateB);
}

Matrix Optimizer::getUpdateA(const Matrix& grad, const Matrix& cur_w, Matrix& memory,
                             Index time_step) const
{
    assert(optimizerA_);
    return optimizerA_(grad, cur_w, memory, time_step);
}

Vector Optimizer::getUpdateB(const Vector& grad, const Vector& cur_w, Vector& memory,
                             Index time_step) const
{
    assert(optimizerB_);
    return optimizerB_(grad, cur_w, memory, time_step);
}

}  // namespace neural_network
