// #include "CrossEntropy.h"

// namespace neural_network
// {
// double CrossEntropy::dist(const Matrix& x, const Matrix& y) const
// {
//     double loss = 0.0;
//     const double epsilon = 1e-15; // To prevent log(0)

//     for (int i = 0; i < x.rows(); ++i) {
//         for (int j = 0; j < x.cols(); ++j) {
//             double predicted = std::clamp(x(i, j), epsilon, 1.0 - epsilon);
//             loss -= y(i, j) * std::log(predicted);
//         }
//     }

//     return loss / y.rows();

// }

// Matrix CrossEntropy::derDist(const Matrix& x, const Matrix& y) const
// {
//     const double epsilon = 1e-15; // To prevent division by zero
//     Matrix gradient = Matrix::Zero(x.rows(), x.cols());

//     for (int i = 0; i < x.rows(); ++i) {
//         for (int j = 0; j < x.cols(); ++j) {
//             double predicted = std::clamp(x(i, j), epsilon, 1.0 - epsilon);
//             gradient(i, j) = -y(i, j) / predicted;
//         }
//     }

//     return gradient;
// }

// } // namespace neural_network;
