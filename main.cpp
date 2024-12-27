#include <Eigen/Dense>
#include <iostream>
#include <random>

int main()
{
    Eigen::MatrixXd matrix(2, 2);
    matrix(0, 0) = 5;
    matrix(0, 1) = 7;
    matrix(1, 0) = 2;
    matrix(1, 1) = 6;
    std::cout << "Исходная матрица:\n" << matrix << std::endl;

    Eigen::MatrixXd transposedMatrix = matrix.transpose();
    std::cout << "Транспонированная матрица:\n" << transposedMatrix << std::endl;

    double determinant = matrix.determinant();
    std::cout << "Детерминант матрицы: " << determinant << std::endl;

    std::mt19937 generator(42);
    std::normal_distribution<double> distribution(0.0,
                                                  1.0); 

    Eigen::MatrixXd randomMatrix(2, 2);
    for (int i = 0; i < randomMatrix.rows(); ++i)
    {
        for (int j = 0; j < randomMatrix.cols(); ++j)
        {
            randomMatrix(i, j) = distribution(generator);
        }
    }
    std::cout << "Случайная матрица:\n" << randomMatrix << std::endl;

    Eigen::MatrixXd resultMatrix = matrix * randomMatrix;
    std::cout << "Результат умножения исходной матрицы на случайную:\n"
              << resultMatrix << std::endl;

    return 0;
}