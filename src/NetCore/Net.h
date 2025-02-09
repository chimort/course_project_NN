#pragma once

#include <memory>
#include <vector>

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

    inline Index getInputSize() const;
    inline Index getOutputSize() const;

private:
    std::vector<DenseLayer> layers_;
};

}  // namespace neural_network
