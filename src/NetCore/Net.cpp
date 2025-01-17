#include <iostream>

#include "Net.h"

namespace neural_network 
{
void Net::compile(std::unique_ptr<Optimizer> optimizer, std::unique_ptr<LossFunction> loss_function,
    std::unique_ptr<ActivationFunction> activation_function)
{
    optimizer_ = std::move(optimizer);
    loss_function_ = std::move(loss_function);
    activation_function_ = std::move(activation_function);
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
            for (const auto& layer: layers_) {
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
} // namespace neural_network