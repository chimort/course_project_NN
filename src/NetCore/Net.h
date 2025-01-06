#pragma once
#include <vector>

class Net
{
public:
    Net(const std::vector<unsigned>& topology);
    void feedForward(const std::vector<double>& inputVals);
    void backProp(const std::vector<double>& targetVals);
    void getResults(std::vector<double>& resultVals) const;

private:
    struct Neuron;
    typedef std::vector<Neuron> Layer;
    std::vector<Layer> m_layers;  // m_layers[layerNum][neuronNum]
};
