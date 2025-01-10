#include "Adam.h"

namespace neural_network
{
Adam::Adam(double learning_rate, double beta1, double beta2, double eps, Index rows, Index cols)
    : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(eps)
{
    m_ = Matrix::Zero(rows, cols);
    v_ = Matrix::Zero(rows, cols);
}

void Adam::updateWeights(Matrix& w, const Matrix& grad_w)
{
    m_ = beta1_ * m_ + (1 - beta1_) * grad_w;
    v_ = beta2_ * v_ + (1 - beta2_) * grad_w.cwiseProduct(grad_w);

    Matrix m_hat = m_ / (1 - beta1_);
    Matrix v_hat = v_ / (1 - beta2_);

    Matrix sqrt_v_hat = v_hat.array().sqrt().matrix();
    w -= learning_rate_ * m_hat.cwiseQuotient(sqrt_v_hat + Matrix::Constant(sqrt_v_hat.rows(), sqrt_v_hat.cols(), eps_));
}
} // namespace neural_network