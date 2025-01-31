#pragma once

#include <memory>
#include <vector>

#include "ActivationFunction.h"
#include "Layer.h"
#include "LossFunction.h"
#include "Math.h"
// #include "Optimizer.h"

// namespace neural_network
// {
// struct LayerParams {
//     int input_size;
//     int output_size;
//     std::shared_ptr<ActivationFunction> activation_function;
// };

// class Net
// {
// public:
//     Net() = default;

//     template <typename LayersType, typename... Args>
//     void addLayer(Args&&... args)
//     {
//         // if (!layers_.empty()) {
//         //     int last_output_size = layers_.back()->getOutputSize();
//         //     int new_input_size = LayersType::getInputSize(std::forward<Args>(args)...);
//         //     assert(last_output_size == new_input_size &&
//         //            "Input size must match the output size of the previous layer!");
//         // }
//         layers_.push_back(std::make_shared<LayersType>(std::forward<Args>(args)...));
//     }

//     void compile(std::unique_ptr<Optimizer> optimizer, std::unique_ptr<LossFunction>
//     loss_function); void fit(const Matrix& df, const Matrix& labels, int epochs, int batch_size);

//     Matrix predict(const Matrix& df) const;

//     double accuracy(const Matrix& df, const Matrix& labels) const;
//     double test_Acc(const Matrix& df, const Matrix& labels) const;

//     inline Index getInputSize() const { return layers_[0]->getInputSize(); }
//     inline Index getOutputSize() const { return layers_.back()->getOutputSize(); }

// private:
//     std::vector<std::shared_ptr<Layer>> layers_;
//     std::unique_ptr<LossFunction> loss_function_;
//     std::unique_ptr<Optimizer> optimizer_;
//     std::unique_ptr<ActivationFunction> activation_function_;
// };

// }  // namespace neural_network
