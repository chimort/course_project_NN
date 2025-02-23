#include "Net.h"

#include <cassert>
#include <iostream>

namespace neural_network
{
void Net::addLayer(std::unique_ptr<DenseLayer> layer) { layers_.push_back(std::move(layer)); }

std::vector<Net::LayersCache> Net::forwardPass(const Matrix& input) const
{
    std::vector<LayersCache> cache_list;
    Matrix output = input;

    for (const auto& layer : layers_) {
        Matrix x = output;
        // Вычисляем линейное преобразование: z = W*x + b
        Matrix z = (layer->getWeights() * x).colwise() + layer->getBiases();
        // Применяем функцию активации к z
        Matrix a = layer->applyActivation(z);

        cache_list.emplace_back(x, a, z);
        output = a;
    }
    return cache_list;
}

void Net::backwardPass(const Matrix& predict, const Matrix& labels,
                       const std::vector<Net::LayersCache>& cache_list, Optimizer& opt,
                       LossFunction& lf, std::vector<Matrix>& weight_memory,
                       std::vector<Vector>& bias_memory, Index epoch)
{
    Matrix error = lf.derDist(predict, labels);

    for (int i = layers_.size() - 1; i >= 0; --i) {
        auto& layer = layers_[i];
        const auto& cache = cache_list[i];  // ← берём кеш для текущего слоя

        Matrix grad_w = layer->getGradW(error, cache.z_, cache.x_);  // ← используем `z`
        Vector grad_b = layer->getGradB(error, cache.z_, cache.x_);

        layer->updateW(opt.getUpdateA(grad_w, layer->getWeights(), weight_memory[i], epoch + 1),
                       weight_memory[i], epoch + 1);
        layer->updateB(opt.getUpdateB(grad_b, layer->getBiases(), bias_memory[i], epoch + 1),
                       bias_memory[i], epoch + 1);

        error = layer->getBackpropError(error, cache.z_, cache.x_);
    }
}

void Net::fit(const Matrix& df, const Matrix& labels, Index epochs, Index batch_size, Optimizer opt,
              LossFunction lf)
{
    DataLoader data_loader(df, labels, batch_size, DataLoader::NormalizeStatus::NotActive);

    std::vector<Matrix> weight_memory(layers_.size());
    std::vector<Vector> bias_memory(layers_.size());

    for (Index epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        Index batch_count = 0;

        std::cout << "\nEpoch " << (epoch + 1) << "/" << epochs << std::endl;

        for (const auto& [batch_data, batch_labels] : data_loader) {
            auto cache_list = forwardPass(batch_data);

            double batch_loss = lf.dist(cache_list.back().activation_,
                                        batch_labels);  // убрать потом
            total_loss += batch_loss;
            std::cout << "  Batch " << (batch_count + 1) << " Loss: " << batch_loss << std::endl;

            backwardPass(cache_list.back().activation_, batch_labels, cache_list, opt, lf,
                         weight_memory, bias_memory, epoch);

            ++batch_count;
        }

        std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                  << " - Loss: " << (total_loss / batch_count) << std::endl;
    }
}

double Net::accuracy(const Matrix& df, const Matrix& labels) const { return 0.0; }

Matrix Net::predict(const Matrix& df) const
{
    Matrix result = df;
    for (const auto& layer : layers_) {
        result = layer->evaluate(result);
    }
    return result;
}

Index Net::getInputSize() const
{
    assert(layers_.size() > 0);
    return layers_[0]->getInputSize();
}

Index Net::getOutputSize() const
{
    assert(layers_.size() > 0);
    return layers_.back()->getOutputSize();
}

}  // namespace neural_network
