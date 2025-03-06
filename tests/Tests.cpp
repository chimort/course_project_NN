#include "Tests.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include "Net.h"

namespace neural_network
{

uint32_t readInt(std::ifstream& file)
{
    uint32_t num = 0;
    file.read(reinterpret_cast<char*>(&num), sizeof(num));
    return __builtin_bswap32(num);
}

Matrix loadMNISTImages(const std::string& filename, int& num_samples, int& rows, int& cols)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Ошибка: не удалось открыть файл " << filename << std::endl;
        exit(1);
    }

    uint32_t magic = readInt(file);
    num_samples = readInt(file);
    rows = readInt(file);
    cols = readInt(file);

    if (magic != 2051) {
        std::cerr << "Ошибка: неверный magic number в " << filename << std::endl;
        exit(1);
    }

    Matrix images(rows * cols, num_samples);

    int image_size = rows * cols;
    for (int i = 0; i < num_samples; i++) {
        std::vector<uint8_t> buffer(image_size);
        file.read(reinterpret_cast<char*>(buffer.data()), image_size);
        std::streamsize bytesRead = file.gcount();
        if (bytesRead != image_size) {
            std::cerr << "Ошибка чтения данных для изображения " << i << ": ожидается "
                      << image_size << " байт, прочитано " << bytesRead << "\n";
            exit(1);
        }

        for (int j = 0; j < image_size; j++) {
            images(j, i) = buffer[j];
        }
    }
    return images;
}

Matrix loadMNISTLabels(const std::string& filename, int& num_samples)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Ошибка: не удалось открыть файл " << filename << std::endl;
        exit(1);
    }

    uint32_t magic = readInt(file);
    num_samples = readInt(file);

    // 2049 is the magic number for label files
    if (magic != 2049) {
        std::cerr << "Ошибка: неверный magic number в " << filename << std::endl;
        exit(1);
    }

    Matrix labels(10, num_samples);
    labels.setZero();

    for (int i = 0; i < num_samples; i++) {
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels(label, i) = 1.0f;
    }
    return labels;
}

template <typename Derived>
int oneHotToLabel(const Eigen::MatrixBase<Derived>& oneHot)
{
    int label = 0;
    oneHot.maxCoeff(&label);
    return label;
}

void test_linear_regression_with_dropout()
{
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
    net.addLayer(AnyLayer::createDenseLayer(In{input_size}, Out{hidden_size}, identityActivation,
                                            optimizer));
    net.addLayer(AnyLayer::createDropoutLayer(In{hidden_size}, Out{hidden_size}, 0.01));
    net.addLayer(AnyLayer::createDenseLayer(In{hidden_size}, Out{output_size}, identityActivation,
                                            optimizer));

    std::cout << "=== До обучения ===\n";
    Matrix pred_before = net.predict(X);
    std::cout << "Предсказания:\n" << pred_before << "\n";

    constexpr int epochs = 100;
    net.fit(X, Y, epochs, num_samples, optimizer, LossFunction::Euclid(),
            Net::NormStatus::NotActive);

    std::cout << "\n=== После обучения (" << epochs << " эпох) ===\n";
    Matrix pred_after = net.predict(X);
    std::cout << "Предсказания:\n" << pred_after << "\n\n";

    double accuracy = net.accuracy(X, Y);
    std::cout << "Точность: " << accuracy << "\n";
}

void test_mnist_classification()
{
    std::cout << "----- Тест: MNIST (классификация цифр 0-9) -----\n";

    constexpr Index input_size = 28 * 28;
    constexpr Index hidden_size = 128;
    constexpr Index output_size = 10;
    int num_train, train_rows, train_cols;

    Matrix X_train = loadMNISTImages("mnist_dataset/train-images.idx3-ubyte", num_train, train_rows,
                                     train_cols);
    Matrix Y_train = loadMNISTLabels("mnist_dataset/train-labels.idx1-ubyte", num_train);

    std::cout << "Тренировочный набор: Загружено " << num_train << " изображений размером "
              << train_rows << "x" << train_cols << "\n";

    ActivationFunction hiddenActivation = ActivationFunction::ReLU();
    ActivationFunction outputActivation = ActivationFunction::Identity();
    Optimizer optimizer = Optimizer::SGD(0.85);

    Net net;
    net.addLayer(
        AnyLayer::createDenseLayer(In{input_size}, Out{hidden_size}, hiddenActivation, optimizer));
    net.addLayer(
        AnyLayer::createDenseLayer(In{hidden_size}, Out{output_size}, outputActivation, optimizer));

    std::cout << "\n=== До обучения (тренировочный набор, первые 5 примеров) ===\n";
    Matrix pred_train_before = net.predict(X_train.leftCols(5));
    std::cout << "Предсказания (вероятностное распределение):\n"
              << pred_train_before.transpose() << "\n";

    std::cout << "Реальные метки: ";
    for (int i = 0; i < 5; ++i) {
        int real_label = oneHotToLabel(Y_train.col(i));
        std::cout << real_label << " ";
    }
    std::cout << "\nПредсказанные метки: ";
    for (int i = 0; i < 5; ++i) {
        int predicted_label = oneHotToLabel(pred_train_before.col(i));
        std::cout << predicted_label << " ";
    }
    std::cout << "\n";

    constexpr int epochs = 10;
    net.fit(X_train, Y_train, epochs, 32, optimizer, LossFunction::Euclid(),
            Net::NormStatus::Active);

    std::cout << "\n=== После обучения (тренировочный набор, первые 5 примеров) ===\n";
    Matrix pred_train_after = net.predict(X_train.leftCols(5));
    std::cout << "Предсказания (вероятностное распределение):\n"
              << pred_train_after.transpose() << "\n";

    std::cout << "Реальные метки: ";
    for (int i = 0; i < 5; ++i) {
        int real_label = oneHotToLabel(Y_train.col(i));
        std::cout << real_label << " ";
    }
    std::cout << "\nПредсказанные метки: ";
    for (int i = 0; i < 5; ++i) {
        int predicted_label = oneHotToLabel(pred_train_after.col(i));
        std::cout << predicted_label << " ";
    }
    std::cout << "\n";

    double train_accuracy = net.accuracy(X_train, Y_train);
    std::cout << "\nТочность на тренировочном наборе: " << train_accuracy << "\n";

    int num_test, test_rows, test_cols;
    Matrix X_test = loadMNISTImages("mnist_dataset/t10k-images.idx3-ubyte", num_test, test_rows,
                                    test_cols);
    Matrix Y_test = loadMNISTLabels("mnist_dataset/t10k-labels.idx1-ubyte", num_test);

    std::cout << "\nТестовый набор: Загружено " << num_test << " изображений размером " << test_rows
              << "x" << test_cols << "\n";

    std::cout << "\n=== Итоговые предсказания (тестовый набор, первые 5 примеров) ===\n";
    Matrix pred_test = net.predict(X_test.leftCols(5));
    std::cout << "Предсказания (вероятностное распределение):\n" << pred_test.transpose() << "\n";

    std::cout << "Реальные метки: ";
    for (int i = 0; i < 5; ++i) {
        int real_label = oneHotToLabel(Y_test.col(i));
        std::cout << real_label << " ";
    }
    std::cout << "\nПредсказанные метки: ";
    for (int i = 0; i < 5; ++i) {
        int predicted_label = oneHotToLabel(pred_test.col(i));
        std::cout << predicted_label << " ";
    }
    std::cout << "\n";

    double test_accuracy = net.accuracy(X_test, Y_test);
    std::cout << "\nТочность на тестовом наборе: " << test_accuracy << "\n";
}

void test_mnist_classification_momentum()
{
    std::cout << "----- Тест: MNIST (классификация цифр 0-9) с оптимизатором momentum -----\n";

    constexpr Index input_size = 28 * 28;
    constexpr Index hidden_size = 128;
    constexpr Index output_size = 10;
    int num_train, train_rows, train_cols;

    Matrix X_train = loadMNISTImages("mnist_dataset/train-images.idx3-ubyte", num_train, train_rows,
                                     train_cols);
    Matrix Y_train = loadMNISTLabels("mnist_dataset/train-labels.idx1-ubyte", num_train);

    std::cout << "Тренировочный набор: Загружено " << num_train << " изображений размером "
              << train_rows << "x" << train_cols << "\n";

    ActivationFunction hiddenActivation = ActivationFunction::ReLU();
    ActivationFunction outputActivation = ActivationFunction::Identity();

    Optimizer momentumOpt = Optimizer::Momentum(0.99, 0.9);

    Net net;
    net.addLayer(AnyLayer::createDenseLayer(In{input_size}, Out{hidden_size}, hiddenActivation,
                                            momentumOpt));
    net.addLayer(AnyLayer::createDenseLayer(In{hidden_size}, Out{output_size}, outputActivation,
                                            momentumOpt));

    std::cout << "\n=== До обучения (тренировочный набор, первые 5 примеров) ===\n";
    Matrix pred_train_before = net.predict(X_train.leftCols(5));
    std::cout << "Предсказания (вероятностное распределение):\n"
              << pred_train_before.transpose() << "\n";

    std::cout << "Реальные метки: ";
    for (int i = 0; i < 5; ++i) {
        int real_label = oneHotToLabel(Y_train.col(i));
        std::cout << real_label << " ";
    }
    std::cout << "\nПредсказанные метки: ";
    for (int i = 0; i < 5; ++i) {
        int predicted_label = oneHotToLabel(pred_train_before.col(i));
        std::cout << predicted_label << " ";
    }
    std::cout << "\n";

    constexpr int epochs = 10;
    net.fit(X_train, Y_train, epochs, 32, momentumOpt, LossFunction::Euclid(),
            Net::NormStatus::Active);

    std::cout << "\n=== После обучения (тренировочный набор, первые 5 примеров) ===\n";
    Matrix pred_train_after = net.predict(X_train.leftCols(5));
    std::cout << "Предсказания (вероятностное распределение):\n"
              << pred_train_after.transpose() << "\n";

    std::cout << "Реальные метки: ";
    for (int i = 0; i < 5; ++i) {
        int real_label = oneHotToLabel(Y_train.col(i));
        std::cout << real_label << " ";
    }
    std::cout << "\nПредсказанные метки: ";
    for (int i = 0; i < 5; ++i) {
        int predicted_label = oneHotToLabel(pred_train_after.col(i));
        std::cout << predicted_label << " ";
    }
    std::cout << "\n";

    double train_accuracy = net.accuracy(X_train, Y_train);
    std::cout << "\nТочность на тренировочном наборе: " << train_accuracy << "\n";

    int num_test, test_rows, test_cols;
    Matrix X_test = loadMNISTImages("mnist_dataset/t10k-images.idx3-ubyte", num_test, test_rows,
                                    test_cols);
    Matrix Y_test = loadMNISTLabels("mnist_dataset/t10k-labels.idx1-ubyte", num_test);

    std::cout << "\nТестовый набор: Загружено " << num_test << " изображений размером " << test_rows
              << "x" << test_cols << "\n";

    std::cout << "\n=== Итоговые предсказания (тестовый набор, первые 5 примеров) ===\n";
    Matrix pred_test = net.predict(X_test.leftCols(5));
    std::cout << "Предсказания (вероятностное распределение):\n" << pred_test.transpose() << "\n";

    std::cout << "Реальные метки: ";
    for (int i = 0; i < 5; ++i) {
        int real_label = oneHotToLabel(Y_test.col(i));
        std::cout << real_label << " ";
    }
    std::cout << "\nПредсказанные метки: ";
    for (int i = 0; i < 5; ++i) {
        int predicted_label = oneHotToLabel(pred_test.col(i));
        std::cout << predicted_label << " ";
    }
    std::cout << "\n";

    double test_accuracy = net.accuracy(X_test, Y_test);
    std::cout << "\nТочность на тестовом наборе: " << test_accuracy << "\n";
}

void test_linear_regression_with_momentum()
{
    std::cout << "----- Тест: Линейная регрессия (y = 3x + 5) с оптимизатором Momentum -----\n";

    constexpr Index input_size = 1;
    constexpr Index hidden_size = 3;
    constexpr Index output_size = 1;
    constexpr Index num_samples = 5;

    Eigen::Matrix<double, input_size, num_samples> X;
    Eigen::Matrix<double, output_size, num_samples> Y;

    X << 0.0, 1.0, 2.0, 3.0, 4.0;
    Y << 5.0, 8.0, 11.0, 14.0, 17.0;

    std::cout << "Входные данные X:\n" << X << "\n\n";
    std::cout << "Целевые данные Y:\n" << Y << "\n\n";

    ActivationFunction activation = ActivationFunction::Identity();

    Optimizer momentumOpt = Optimizer::Momentum(0.5, 0.9);
    Net netMomentum;
    netMomentum.addLayer(
        AnyLayer::createDenseLayer(In{input_size}, Out{hidden_size}, activation, momentumOpt));
    netMomentum.addLayer(
        AnyLayer::createDenseLayer(In{hidden_size}, Out{output_size}, activation, momentumOpt));

    std::cout << "=== (Momentum) До обучения ===\n";
    Eigen::MatrixXd pred_before = netMomentum.predict(X);
    std::cout << "Предсказания:\n" << pred_before << "\n";
    constexpr int epochs = 100;
    netMomentum.fit(X, Y, epochs, num_samples, momentumOpt, LossFunction::Euclid(),
                    Net::NormStatus::NotActive);

    std::cout << "\n=== (Momentum) После обучения (" << epochs << " эпох) ===\n";
    Eigen::MatrixXd pred_after = netMomentum.predict(X);
    std::cout << "Предсказания:\n" << pred_after << "\n\n";

    double accuracy = netMomentum.accuracy(X, Y);
    std::cout << "Точность (Momentum): " << accuracy << "\n\n";
}

void test_linear_regression_with_adam()
{
    std::cout << "----- Тест: Линейная регрессия (y = 3x + 5) с оптимизатором Adam -----\n";

    constexpr Index input_size = 1;
    constexpr Index hidden_size = 3;
    constexpr Index output_size = 1;
    constexpr Index num_samples = 5;

    Eigen::Matrix<double, input_size, num_samples> X;
    Eigen::Matrix<double, output_size, num_samples> Y;
    X << 0.0, 1.0, 2.0, 3.0, 4.0;
    Y << 5.0, 8.0, 11.0, 14.0, 17.0;

    std::cout << "Входные данные X:\n" << X << "\n\n";
    std::cout << "Целевые данные Y:\n" << Y << "\n\n";

    ActivationFunction activation = ActivationFunction::Identity();

    Optimizer adamOpt = Optimizer::Adam(0.5, 0.9, 0.999, 1e-8);
    Net netAdam;
    netAdam.addLayer(
        AnyLayer::createDenseLayer(In{input_size}, Out{hidden_size}, activation, adamOpt));
    netAdam.addLayer(
        AnyLayer::createDenseLayer(In{hidden_size}, Out{output_size}, activation, adamOpt));

    std::cout << "=== (Adam) До обучения ===\n";
    Eigen::MatrixXd pred_before = netAdam.predict(X);
    std::cout << "Предсказания:\n" << pred_before << "\n";

    constexpr int epochs = 100;
    netAdam.fit(X, Y, epochs, num_samples, adamOpt, LossFunction::Euclid(),
                Net::NormStatus::NotActive);

    std::cout << "\n=== (Adam) После обучения (" << epochs << " эпох) ===\n";
    Eigen::MatrixXd pred_after = netAdam.predict(X);
    std::cout << "Предсказания:\n" << pred_after << "\n\n";

    double accuracy = netAdam.accuracy(X, Y);
    std::cout << "Точность (Adam): " << accuracy << "\n\n";
}

void run_all_tests()
{
    test_linear_regression_with_dropout();
    // test_mnist_classification();
    // test_mnist_classification_momentum();
    test_linear_regression_with_momentum();
    test_linear_regression_with_adam();
}

}  // namespace neural_network
