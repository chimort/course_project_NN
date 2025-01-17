#pragma once

#include <memory>
#include <vector>

#include "ActivationFunction.h"
#include "Config.h"
#include "Layer.h"
#include "LossFunction.h"
#include "Optimizer.h"

namespace neural_network
{

struct LayerParams {
    int input_size;
    int output_size;
    std::shared_ptr<ActivationFunction> activation_function;
};

class Net
{
public:
    Net() = default;

    // это потом в какой-нибудь адаптер вынести
    template <typename LayersType>
    void addLayer(const LayerParams& config);

    template <typename... LayersConfig>
    inline void addLayers(LayersConfig&&... configs) {
        (addLayer<DenseLayer>(std::forward<LayersConfig>(configs)), ...);
    }

    void compile(std::unique_ptr<Optimizer> optimizer, std::unique_ptr<LossFunction> loss_function);
    void fit(const Matrix& df, const Matrix& labels, int epochs, int batch_size);

    Matrix predict(const Matrix& df) const;

    double accuracy(const Matrix& df, const Matrix& labels) const;

    inline Index getInputSize() const { return layers_[0]->getInputSize(); }
    inline Index getOutputSize() const { return layers_.back()->getOutputSize(); }

private:
    std::vector<std::shared_ptr<Layer>> layers_;
    std::unique_ptr<LossFunction> loss_function_;
    std::unique_ptr<Optimizer> optimizer_;
    std::unique_ptr<ActivationFunction> activation_function_;
};

}  // namespace neural_network
