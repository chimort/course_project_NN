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
        return -learning_rate * grad;
    };

    VSignature sgdUpdateB = [learning_rate](const Vector& grad, const Vector& currentBiases,
                                            Vector& memory, int time_step) -> Vector {
        return -learning_rate * grad;
    };

    return Optimizer(sgdUpdateA, sgdUpdateB);
}

// Optimizer Optimizer::Momentum(double learning_rate, double beta1)
// {
//     auto m = std::make_shared<Matrix>();

//     return Optimizer([m, learning_rate, beta1](const Matrix& grad, Matrix* weights) {
//         if (m->rows() != grad.rows() || m->cols() != grad.cols()) {
//             *m = Matrix::Zero(grad.rows(), grad.cols());
//         }

//         *m = beta1 * (*m) + (1 - beta1) * grad;

//         *weights -= learning_rate * (*m);
//     });
// }

// Optimizer Optimizer::Adam(double learning_rate, double beta1, double beta2, double epsilon)
// {
//     auto m = std::make_shared<Matrix>();
//     auto v = std::make_shared<Matrix>();
//     auto t = std::make_shared<int>(0);

//     return Optimizer([m, v, t, learning_rate, beta1, beta2, epsilon](const Matrix& grad,
//                                                                      Matrix* weights) mutable {
//         if (m->rows() != grad.rows() || m->cols() != grad.cols()) {
//             *m = Matrix::Zero(grad.rows(), grad.cols());
//             *v = Matrix::Zero(grad.rows(), grad.cols());
//         }

//         ++(*t);
//         *m = beta1 * (*m) + (1 - beta1) * grad;
//         *v = beta2 * (*v) + (1 - beta2) * grad.array().square().matrix();
//         Matrix mHat = (*m) / (1 - std::pow(beta1, *t));
//         Matrix vHat = (*v) / (1 - std::pow(beta2, *t));
//         *weights -= (learning_rate * mHat.array() / (vHat.array().sqrt() + epsilon)).matrix();
//     });
// }

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
