#include <Eigen/Dense>
#include <iostream>

#include "ActivationFunction.h"
#include "Layer.h"

using namespace neural_network;
using namespace Eigen;

int main()
{
    // Размеры входных и выходных данных
    int input_size = 3;
    int output_size = 2;

    // Выбор активационной функции
    ReLU relu_activation;

    // Создание слоя
    Layer layer(input_size, output_size, &relu_activation);

    // Пример входных данных
    MatrixXd input = MatrixXd::Random(input_size, 5); 

    MatrixXd output = layer.evaluate(input);
    std::cout << "Output of the layer: \n" << output << std::endl;

    MatrixXd a = MatrixXd::Random(output_size, 5); 
    MatrixXd b = MatrixXd::Random(input_size, 5);  

    MatrixXd gradW = layer.getGradW(a, b);
    MatrixXd gradB = layer.getGradB(a, b);

    std::cout << "Gradient W: \n" << gradW << std::endl;
    std::cout << "Gradient B: \n" << gradB << std::endl;

    // Обновление весов и смещений
    layer.updateW(gradW);
    layer.updateB(gradB);

    return 0;
}