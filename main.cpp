#include <iostream>

#include "LossFunction.h"

int main()
{
    using namespace neural_network;

    EuclidDist distFunc;
    Eigen::MatrixXd x(2, 2);
    Eigen::MatrixXd y(2, 2);

    x << 1, 2, 3, 4;
    y << 1, 1, 1, 1;

    // Вычисление расстояния
    double distance = distFunc.dist(x, y);
    std::cout << "Euclidean Distance: " << distance << std::endl;

    return 0;
}
