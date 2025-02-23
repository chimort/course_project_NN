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
    void addLayer(std::unique_ptr<DenseLayer> layer);
    void fit(const Matrix& df, const Matrix& labels, Index epochs, Index batch_size, Optimizer opt,
             LossFunction lf);

    Matrix predict(const Matrix& df) const;

    double accuracy(const Matrix& df, const Matrix& labels) const;

    Index getInputSize() const;
    Index getOutputSize() const;

private:
    struct LayersCache {
        Matrix x_;
        Matrix activation_;
        Matrix z_;

        LayersCache(const Matrix& x, const Matrix& activation, const Matrix& z)
            : x_(x), activation_(activation), z_(z)
        {
        }
    };

    std::vector<std::unique_ptr<DenseLayer>> layers_;

    std::vector<LayersCache> forwardPass(const Matrix& input) const;

    void backwardPass(const Matrix& predict, const Matrix& labels,
                      const std::vector<LayersCache>& cache_list, Optimizer& opt, LossFunction& lf,
                      std::vector<Matrix>& weight_memory, std::vector<Vector>& bias_memory,
                      Index epoch);
};

}  // namespace neural_network
