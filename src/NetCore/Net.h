#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "DataLoader.h"
#include "DenseLayer.h"
#include "LossFunction.h"
#include "Math.h"
#include "Optimizer.h"

namespace neural_network
{
class Net
{
public:
    void addLayer(DenseLayer layer);
    void fit(const Matrix& df, const Matrix& labels, Index epochs, Index batch_size, Optimizer opt,
             LossFunction lf);

    Matrix predict(const Matrix& df) const;

    double accuracy(const Matrix& df, const Matrix& labels) const;

    Index getInputSize() const;
    Index getOutputSize() const;

private:
    std::vector<DenseLayer> layers_;

    std::pair<Matrix, std::vector<Matrix>> forwardPass(const Matrix& input) const;

    void backwardPass(const Matrix& predict, const Matrix& labels,
                      const std::vector<Matrix>& activations, Optimizer& opt, LossFunction& lf,
                      std::vector<Matrix>& weight_memory, std::vector<Vector>& bias_memory,
                      Index epoch);
};

}  // namespace neural_network
