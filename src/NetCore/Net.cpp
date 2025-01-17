#include "Net.h"

#include <iostream>

namespace neural_network
{
template <typename LayersType>
void Net::addLayer(const LayerParams& config) {
    if (!layers_.empty()) {
        int last_output_size = layers_.back()->getOutputSize();
        assert(last_output_size == config.input_size &&
                "Input size must match the output size of the previous layer!");
    }
    layers_.push_back(std::make_shared<LayersType>(
        config.input_size,
        config.output_size,
        config.activation_function
    ));
}

void Net::compile(std::unique_ptr<Optimizer> optimizer, std::unique_ptr<LossFunction> loss_function)
{
    optimizer_ = std::move(optimizer);
    loss_function_ = std::move(loss_function);
    assert(layers_.size() >= 2 && "Network must have at least two layers!");
}

void Net::fit(const Matrix& df, const Matrix& labels, int epochs, int batch_size)
{
    int num_samples = df.rows();
    int steps_per_epoch = num_samples / batch_size;

    for (int i = 1; i <= epochs; ++i) {
        double epoch_loss = 0.0;

        for (int j = 0; j < steps_per_epoch; ++j) {
            int start_idx = batch_size * j;
            int end_idx = std::min(start_idx + batch_size, num_samples);
            Matrix batch_data = df.block(start_idx, 0, end_idx - start_idx, df.cols());
            Matrix batch_labels = labels.block(start_idx, 0, end_idx - start_idx, labels.cols());

            Matrix output = batch_data;
            for (const auto& layer : layers_) {
                output = layer->evaluate(output);
            }

            epoch_loss += loss_function_->dist(output, batch_labels);

            Matrix grad = loss_function_->derDist(output, batch_labels);
            for (int k = layers_.size() - 1; k >= 0; --k) {
                grad = layers_[k]->getBackpropError(grad, batch_data);

                Matrix grad_w = layers_[k]->getGradW(grad, batch_labels);
                Matrix grad_b = layers_[k]->getGradB(grad, batch_labels);

                layers_[k]->updateW(grad_w);
                layers_[k]->updateB(grad_b);
            }
            optimizer_->updateWeights(batch_data, grad);
        }
        // Move this info to somewhere else mb
        double avg_loss = epoch_loss / steps_per_epoch;
        std::cout << "Epoch [" << i << "/" << epochs << "] - Loss: " << avg_loss << std::endl;
    }
}

Matrix Net::predict(const Matrix& df) const
{
    Matrix output = df;

    for (const auto& layer : layers_) {
        output = layer->evaluate(output);
    }

    return output;
}

double Net::accuracy(const Matrix& df, const Matrix& labels) const
{
    Matrix output = predict(df);
    Matrix predicted_labels = output.rowwise().maxCoeff();
    Matrix correct_predictions = (predicted_labels.array() == labels.array()).cast<double>();

    return correct_predictions.mean();
}

}  // namespace neural_network