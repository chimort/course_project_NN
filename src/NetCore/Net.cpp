#include "Net.h"

#include <cassert>
#include <iostream>

namespace neural_network
{
void Net::addLayer(DenseLayer layer) { layers_.push_back(layer); }

std::pair<Matrix, std::vector<Matrix>> Net::forwardPass(const Matrix& input) const
{
    std::vector<Matrix> activations;
    Matrix output = input;
    for (const auto& layer : layers_) {
        activations.push_back(output);
        output = layer.evaluate(output);
    }
    return {output, activations};
}

void Net::backwardPass(const Matrix& predict, const Matrix& labels,
                       const std::vector<Matrix>& activations, Optimizer& opt, LossFunction& lf,
                       std::vector<Matrix>& weight_memory, std::vector<Vector>& bias_memory,
                       Index epoch)
{
    Matrix error = lf.derDist(predict, labels);
    for (int i = layers_.size() - 1; i >= 0; --i) {
        auto& layer = layers_[i];

        Matrix grad_w = layer.getGradW(error, activations[i]);

        // std::cout << "Grad W avg: " << grad_w.mean() << " min: " << grad_w.minCoeff()
        //           << " max: " << grad_w.maxCoeff() << "\n";

        Vector grad_b = layer.getGradB(error, activations[i]);

        layer.updateW(opt.getUpdateA(grad_w, layer.getWeights(), weight_memory[i], epoch + 1),
                      weight_memory[i], epoch + 1);
        layer.updateB(opt.getUpdateB(grad_b, layer.getBiases(), bias_memory[i], epoch + 1),
                      bias_memory[i], epoch + 1);

        error = layer.getBackpropError(error, activations[i]);
    }
}

void Net::fit(const Matrix& df, const Matrix& labels, Index epochs, Index batch_size, Optimizer opt,
              LossFunction lf)
{
    DataLoader data_loader(df, labels, batch_size);

    std::vector<Matrix> weight_memory(layers_.size());
    std::vector<Vector> bias_memory(layers_.size());

    for (Index epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        Index batch_count = 0;

        for (const auto& [batch_data, batch_labels] : data_loader) {
            auto [predict, activations] = forwardPass(batch_data);

            total_loss += lf.dist(predict, batch_labels);

            backwardPass(predict, batch_labels, activations, opt, lf, weight_memory, bias_memory,
                         epoch);

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
        result = layer.evaluate(result);
    }
    return result;
}

Index Net::getInputSize() const
{
    assert(layers_.size() > 0);
    return layers_[0].getInputSize();
}

Index Net::getOutputSize() const
{
    assert(layers_.size() > 0);
    return layers_.back().getOutputSize();
}

}  // namespace neural_network
