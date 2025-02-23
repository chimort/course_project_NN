#include "Net.h"

#include <cassert>
#include <iostream>

namespace neural_network
{
void Net::addLayer(std::unique_ptr<DenseLayer> layer) { layers_.push_back(std::move(layer)); }

std::vector<Net::TrainCache> Net::inicializeCache(int size)
{
    std::vector<TrainCache> cache_list;
    cache_list.reserve(size);
    for (int i = 0; i < size; ++i) {
        TrainCache cache;
        cache.inicializeMemory(size, size);
        cache_list.push_back(std::move(cache));
    }

    return cache_list;
}

void Net::forwardPass(const Matrix& input, std::vector<TrainCache>& cache_list) const
{
    Matrix output = input;

    for (int i = 0; i < cache_list.size(); ++i) {
        const auto& layer = layers_[i];
        Matrix x = output;
        Matrix z = (layer->getWeights() * x).colwise() + layer->getBiases();
        Matrix a = layer->applyActivation(z);

        cache_list[i].x_ = x;
        cache_list[i].activation_ = a;
        cache_list[i].z_ = z;
        output = a;
    }
}

void Net::backwardPass(const Matrix& predict, const Matrix& labels,
                       std::vector<Net::TrainCache>& cache_list, Optimizer& opt, LossFunction& lf,
                       Index epoch)
{
    Matrix error = lf.derDist(predict, labels);

    for (int i = layers_.size() - 1; i >= 0; --i) {
        auto& layer = layers_[i];
        auto& cache = cache_list[i];

        Matrix grad_w = layer->getGradW(error, cache.z_, cache.x_);
        Vector grad_b = layer->getGradB(error, cache.z_, cache.x_);

        layer->updateW(opt.getUpdateA(grad_w, layer->getWeights(), cache.weight_memory_[i],
                                      epoch + 1),
                       cache.weight_memory_[i], epoch + 1);
        layer->updateB(opt.getUpdateB(grad_b, layer->getBiases(), cache.bias_memory_[i], epoch + 1),
                       cache.bias_memory_[i], epoch + 1);

        error = layer->getBackpropError(error, cache.z_, cache.x_);
    }
}

void Net::fit(const Matrix& df, const Matrix& labels, Index epochs, Index batch_size, Optimizer opt,
              LossFunction lf)
{
    DataLoader data_loader(df, labels, batch_size, DataLoader::NormalizeStatus::NotActive);

    std::vector<TrainCache> cache_list = inicializeCache(layers_.size());

    for (Index epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        Index batch_count = 0;

        for (const auto& [batch_data, batch_labels] : data_loader) {
            forwardPass(batch_data, cache_list);

            total_loss += lf.dist(cache_list.back().activation_, batch_labels);
            backwardPass(cache_list.back().activation_, batch_labels, cache_list, opt, lf, epoch);

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
