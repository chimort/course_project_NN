#include "Net.h"

#include <cassert>
#include <iostream>

namespace neural_network
{
void Net::addLayer(DenseLayer layer) { layers_.push_back(layer); }

void Net::fit(const Matrix& df, const Matrix& lables, Index epochs, Index batch_size, Optimizer opt,
              LossFunction lf)
{
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
