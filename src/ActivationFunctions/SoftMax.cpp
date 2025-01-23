#include "SoftMax.h"

namespace neural_network
{
double SoftMax::evaluate(const double& x) const { return 1.0; }
double SoftMax::derEvaluate(const double& x) const { return 0.0; }

Matrix SoftMax::evaluate(const Matrix& x) const
{
    Matrix res(x.rows(), x.cols());

    for (int i = 0; i < x.rows(); ++i) {
        double max_coeff = x.row(i).maxCoeff();
        Matrix row = x.row(i).array() - max_coeff;
        Matrix exp_row = row.array().exp();
        double sum_exp = exp_row.sum();
        res.row(i) = exp_row / sum_exp;
    }

    return res;
}
Matrix SoftMax::derEvaluate(const Matrix& x) const
{
    Matrix softmax = evaluate(x);
    Matrix der = Matrix::Zero(x.rows(), x.cols());

    for (int i = 0; i < softmax.rows(); ++i) {
        for (int j = 0; j < softmax.cols(); ++j) {
            if (i == j) {
                der(i, j) = softmax(i, j) * (1 - softmax(i, j));
            } else {
                der(i, j) = -softmax(i, j) * softmax(j, j);
            }
        }
    }

    return der;
}

}  // namespace neural_network
