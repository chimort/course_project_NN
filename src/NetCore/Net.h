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
        layers.push_back(std::make_shared<LayersType>(std::forward<Args>(args)...));
    }

    void compile(std::unique_ptr<Optimizer> optimizer, std::unique_ptr<LossFunction> loss_funtion,
                 std::unique_ptr<ActivationFunction> activation_function);
    void fit(const Matrix& df, int epochs, int batch_size);

    Matrix predict(const Matrix& df) const;

    double accuracy(const Matrix& df) const;

    Index getInputSize() const;
    Index getOutputSize() const;

private:
    std::vector<std::shared_ptr<Layer>> layers_;
    std::unique_ptr<LossFunction> loss_function_;
    std::unique_ptr<Optimizer> optimizer_;
    std::unique_ptr<ActivationFunction> activation_function_;
};
}  // namespace neural_network
