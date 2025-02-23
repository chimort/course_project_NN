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
    struct TrainCache {
        Matrix x_;
        Matrix activation_;
        Matrix z_;

        std::vector<Matrix> weight_memory_;
        std::vector<Vector> bias_memory_;

        void inicializeMemory(int weights_size, int biases_size)
        {
            weight_memory_.resize(weights_size);
            bias_memory_.resize(biases_size);
        }
    };

    std::vector<TrainCache> inicializeCache(int size);

    std::vector<std::unique_ptr<DenseLayer>> layers_;

    void forwardPass(const Matrix& input, std::vector<TrainCache>& cache_list) const;

    void backwardPass(const Matrix& predict, const Matrix& labels,
                      std::vector<TrainCache>& cache_list, Optimizer& opt, LossFunction& lf,
                      Index epoch);
};

}  // namespace neural_network
