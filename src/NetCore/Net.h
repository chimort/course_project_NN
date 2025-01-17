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
class Net
{
public:
    Net();

    template <typename LayersType, typename... Args>
    inline void addLayer(Args&&... args)
    {
        layers_.push_back(std::make_shared<LayersType>(std::forward<Args>(args)...));
    }

    void compile(std::unique_ptr<Optimizer> optimizer, std::unique_ptr<LossFunction> loss_function,
                 std::unique_ptr<ActivationFunction> activation_function);
    void fit(const Matrix& df, const Matrix& labels, int epochs, int batch_size);

    Matrix predict(const Matrix& df) const;

    double accuracy(const Matrix& df) const;

    inline Index getInputSize() const { return layers_[0]->getInputSize(); }
    inline Index getOutputSize() const { return layers_.back()->getOutputSize(); }

private:
    std::vector<std::shared_ptr<Layer>> layers_;
    std::unique_ptr<LossFunction> loss_function_;
    std::unique_ptr<Optimizer> optimizer_;
    std::unique_ptr<ActivationFunction> activation_function_;
};
}  // namespace neural_network
