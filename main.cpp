#include <Eigen/Dense>
#include <iostream>

#include "ActivationFunction.h"
#include "DenseLayer.h"
#include "DropoutLayer.h"
#include "ReLU.h"

using namespace neural_network;
using namespace Eigen;

int main()
{
    // Размеры входных и выходных данных
    int input_size = 3;
    int output_size = 2;

    // Выбор активационной функции
    ReLU relu_activation;

    // Создание слоя Dense
    DenseLayer dense_layer(input_size, output_size, std::make_unique<ReLU>(relu_activation));

    // Пример входных данных
    MatrixXd input = MatrixXd::Random(input_size, 5);  // 5 примеров

    // Вывод результатов для DenseLayer
    MatrixXd output = dense_layer.evaluate(input);
    std::cout << "Output of DenseLayer: \n" << output << std::endl;

    MatrixXd a = MatrixXd::Random(output_size, 5);
    MatrixXd b = MatrixXd::Random(input_size, 5);

    // Получение градиентов для DenseLayer
    MatrixXd gradW = dense_layer.getGradW(a, b);
    MatrixXd gradB = dense_layer.getGradB(a, b);

    std::cout << "Gradient W for DenseLayer: \n" << gradW << std::endl;
    std::cout << "Gradient B for DenseLayer: \n" << gradB << std::endl;

    // Обновление весов и смещений для DenseLayer
    dense_layer.updateW(gradW);
    dense_layer.updateB(gradB);

    // Создание слоя Dropout
    DropoutLayer dropout_layer(0.5);  // rate = 50%

    // Установка режима обучения
    dropout_layer.setTrainingMode(true);

    // Применение Dropout
    MatrixXd dropout_output = dropout_layer.evaluate(input);
    std::cout << "Output of DropoutLayer (in training mode): \n" << dropout_output << std::endl;

    // Переключение на режим тестирования
    dropout_layer.setTrainingMode(false);

    // Применение Dropout снова, но в тестовом режиме
    dropout_output = dropout_layer.evaluate(input);
    std::cout << "Output of DropoutLayer (in testing mode): \n" << dropout_output << std::endl;

    return 0;
}
