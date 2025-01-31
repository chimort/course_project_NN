// #include "Net.h"

// #include <iostream>

// namespace neural_network
// {
// void Net::compile(std::unique_ptr<Optimizer> optimizer, std::unique_ptr<LossFunction>
// loss_function)
// {
//     optimizer_ = std::move(optimizer);
//     loss_function_ = std::move(loss_function);
//     assert(layers_.size() >= 2 && "Network must have at least two layers!");
// }

// void Net::fit(const Matrix& df, const Matrix& labels, int epochs, int batch_size)
// {
//     assert(df.rows() == labels.rows() && "Number of samples must match number of labels");

//     int num_samples = df.rows();
//     int steps_per_epoch = num_samples / batch_size;

//     for (int i = 1; i <= epochs; ++i) {
//         double epoch_loss = 0.0;

//         for (int j = 0; j < steps_per_epoch; ++j) {
//             // Правильное вычисление индексов батча
//             int start_idx = j * batch_size;
//             int actual_batch_size = std::min(batch_size, num_samples - start_idx);

//             // Извлечение батча данных
//             Matrix batch_data = df.block(start_idx, 0, actual_batch_size, df.cols());
//             Matrix batch_labels = labels.block(start_idx, 0, actual_batch_size, labels.cols());

//             // Forward pass с сохранением выходов каждого слоя
//             std::vector<Matrix> layer_outputs;
//             layer_outputs.reserve(layers_.size() + 1);  // +1 для входного слоя
//             layer_outputs.push_back(batch_data);  // Сохраняем входные данные

//             Matrix current_output = batch_data;
//             for (const auto& layer : layers_) {
//                 current_output = layer->evaluate(current_output);
//                 layer_outputs.push_back(current_output);
//             }

//             // Вычисление функции потерь
//             epoch_loss += loss_function_->dist(current_output, batch_labels);

//             // Backward pass
//             Matrix grad = loss_function_->derDist(current_output, batch_labels);

//             // Обратное распространение ошибки через слои
//             for (int k = layers_.size() - 1; k >= 0; --k) {
//                 const Matrix& layer_input = layer_outputs[k];

//                 // Вычисление градиентов
//                 Matrix grad_w = layers_[k]->getGradW(grad, layer_input);
//                 Matrix grad_b = layers_[k]->getGradB(grad, layer_input);

//                 // Обновление весов и смещений
//                 layers_[k]->updateW(grad_w);
//                 layers_[k]->updateB(grad_b);

//                 // Вычисление градиента для следующего слоя
//                 if (k > 0) {  // Не нужно вычислять для входного слоя
//                     grad = layers_[k]->getBackpropError(grad, layer_input);
//                 }
//             }
//         }

//         // Вывод информации о процессе обучения
//         double avg_loss = epoch_loss / steps_per_epoch;
//         std::cout << "Epoch [" << i << "/" << epochs << "] - Loss: " << avg_loss << std::endl;
//     }
// }

// Matrix Net::predict(const Matrix& df) const
// {
//     Matrix output = df;

//     for (const auto& layer : layers_) {
//         output = layer->evaluate(output);
//     }

//     return output;
// }

// double Net::accuracy(const Matrix& df, const Matrix& labels) const
// {
//     Matrix output = predict(df);

//     // Вектор индексов предсказанных классов
//     Eigen::VectorXi predicted_indices(output.rows());
//     for (int i = 0; i < output.rows(); ++i) {
//         double max_val;
//         Eigen::Index max_index;
//         output.row(i).maxCoeff(&max_index);
//         predicted_indices(i) = max_index;
//     }

//     // Вектор индексов истинных классов
//     Eigen::VectorXi true_indices(labels.rows());
//     for (int i = 0; i < labels.rows(); ++i) {
//         double val;
//         Eigen::Index index;
//         labels.row(i).maxCoeff(&index);
//         true_indices(i) = index;
//     }

//     // Сравниваем предсказанные и истинные классы
//     int correct = 0;
//     for (int i = 0; i < predicted_indices.size(); ++i) {
//         if (predicted_indices(i) == true_indices(i)) correct++;
//     }

//     return static_cast<double>(correct) / output.rows();
// }

// double Net::test_Acc(const Matrix& df, const Matrix& labels) const
// {
//     Matrix output = predict(df);  // Предсказания сети (вероятности)

//     // Преобразуем вероятности в бинарные предсказания (0 или 1)
//     Matrix predicted_labels = (output.array() >= 0.5).cast<double>();

//     // Сравниваем предсказания с истинными метками
//     Matrix correct_predictions = (predicted_labels.array() == labels.array()).cast<double>();

//     // Возвращаем долю правильных предсказаний
//     return correct_predictions.sum() / labels.rows();
// }

// }  // namespace neural_network
