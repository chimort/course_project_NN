#include "Tests.h"

#include <Eigen/Dense>
#include <iostream>

#include "Net.h"


namespace neural_network {

void test_linear_regression_with_dropout() {
    std::cout << "----- Тест: Линейная регрессия (y = 2x + 1) с DropoutLayer -----\n";

    constexpr Index input_size = 1;
    constexpr Index hidden_size = 3;
    constexpr Index output_size = 1;
    constexpr Index num_samples = 5;

    Matrix X(input_size, num_samples);
    Matrix Y(output_size, num_samples);
    X << 0.0, 1.0, 2.0, 3.0, 4.0;
    Y << 1.0, 3.0, 5.0, 7.0, 9.0;

    std::cout << "Входные данные X:\n" << X << "\n\n";
    std::cout << "Целевые данные Y:\n" << Y << "\n\n";

    ActivationFunction identityActivation = ActivationFunction::Identity();
    Optimizer optimizer = Optimizer::SGD(0.5);

    Net net;
    net.addLayer(AnyLayer::createDenseLayer(In{input_size}, Out{hidden_size}, identityActivation, optimizer));
    net.addLayer(AnyLayer::createDropoutLayer(In{hidden_size}, Out{hidden_size}, 0.01));
    net.addLayer(AnyLayer::createDenseLayer(In{hidden_size}, Out{output_size}, identityActivation, optimizer));

    std::cout << "=== До обучения ===\n";
    Matrix pred_before = net.predict(X);
    std::cout << "Предсказания:\n" << pred_before << "\n";

    constexpr int epochs = 500;
    net.fit(X, Y, epochs, num_samples, optimizer, LossFunction::Euclid(), Net::NormStatus::NotActive);

    std::cout << "\n=== После обучения (" << epochs << " эпох) ===\n";
    Matrix pred_after = net.predict(X);
    std::cout << "Предсказания:\n" << pred_after << "\n\n";

    double accuracy = net.accuracy(X, Y);
    std::cout << "Точность: " << accuracy << "\n";
}

void run_all_tests() {
    test_linear_regression_with_dropout();
}

} // namespace neural_network
