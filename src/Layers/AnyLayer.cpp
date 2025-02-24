#include "AnyLayer.h"

namespace neural_network {

template<typename, typename = std::void_t<>>
struct HasGetGradW : std::false_type {};

template<typename T>
struct HasGetGradW<T, std::void_t<decltype(std::declval<T>().getGradW(
                        std::declval<Matrix>(), std::declval<Matrix>(), std::declval<Matrix>()))>>
    : std::true_type {};

template<typename, typename = std::void_t<>>
struct HasGetGradB : std::false_type {};

template<typename T>
struct HasGetGradB<T, std::void_t<decltype(std::declval<T>().getGradB(
                        std::declval<Matrix>(), std::declval<Matrix>(), std::declval<Matrix>()))>>
    : std::true_type {};

template<typename, typename = std::void_t<>>
struct HasGetBackpropError : std::false_type {};

template<typename T>
struct HasGetBackpropError<T, std::void_t<decltype(std::declval<T>().getBackpropError(
                        std::declval<Matrix>(), std::declval<Matrix>()))>>
    : std::true_type {};

template <typename T, typename = void>
struct HasGetWeights : std::false_type {};

template <typename T>
struct HasGetWeights<T, std::void_t<decltype(std::declval<T>().getWeights())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetBiases : std::false_type {};

template <typename T>
struct HasGetBiases<T, std::void_t<decltype(std::declval<T>().getBiases())>> : std::true_type {};

template <typename T, typename = void>
struct HasUpdateW : std::false_type {};

template <typename T>
struct HasUpdateW<T, std::void_t<decltype(std::declval<T>().updateW(std::declval<Matrix>(), std::declval<Matrix&>(), std::declval<int>()))>> : std::true_type {};

template <typename T, typename = void>
struct HasUpdateB : std::false_type {};

template <typename T>
struct HasUpdateB<T, std::void_t<decltype(std::declval<T>().updateB(std::declval<Vector>(), std::declval<Vector&>(), std::declval<int>()))>> : std::true_type {};

template <typename T, typename = void>
struct HasGetInputSize : std::false_type {};

template <typename T>
struct HasGetInputSize<T, std::void_t<decltype(std::declval<T>().getInputSize())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetOutputSize : std::false_type {};

template <typename T>
struct HasGetOutputSize<T, std::void_t<decltype(std::declval<T>().getOutputSize())>> : std::true_type {};

AnyLayer::AnyLayer(LayerType layer) : layer_(std::move(layer)) {}

AnyLayer AnyLayer::createDenseLayer(In in_size, Out out_size, ActivationFunction f, Optimizer opt)
{
    return AnyLayer(DenseLayer(in_size, out_size, f, opt));
}

AnyLayer AnyLayer::createDropoutLayer(In in_size, Out out_size, double rate)
{
    return AnyLayer(DropoutLayer(in_size, out_size, rate));
}

Matrix AnyLayer::evaluate(const Matrix& input) const
{
    return std::visit([&](const auto& l) -> Matrix {
        return l.evaluate(input);
    }, layer_);
}

Matrix AnyLayer::getGradW(const Matrix& a, const Matrix& z, const Matrix& b) const
{
    return std::visit([&](const auto& l) -> Matrix {
        if constexpr (HasGetGradW<decltype(l)>::value) {
            return l.getGradW(a, z, b);
        } else {
            return {};
        }
    }, layer_);
}

Matrix AnyLayer::getGradB(const Matrix& a, const Matrix& z, const Matrix& b) const
{
    return std::visit([&](const auto& l) -> Matrix {
        if constexpr (HasGetGradB<decltype(l)>::value) {
            return l.getGradB(a, z, b);
        } else {
            return {};
        }
    }, layer_);
}

Matrix AnyLayer::getBackpropError(const Matrix& a, const Matrix& z, const Matrix& b) const
{
    return std::visit([&](const auto& l) -> Matrix {
        if constexpr (HasGetBackpropError<decltype(l)>::value) {
            return l.getBackpropError(a, z, b);
        } else {
            return a;
        }
    }, layer_);
}

bool AnyLayer::hasWeights() const {
    return std::visit([](const auto& l) -> bool {
        return HasGetWeights<decltype(l)>::value;
    }, layer_);
}

void AnyLayer::updateW(const Matrix& grad_diff, Matrix& memory, int time_step) {
    std::visit([&](auto& l) {
        if constexpr (HasUpdateW<decltype(l)>::value) {
            l.updateW(grad_diff, memory, time_step);
        }
    }, layer_);
}

void AnyLayer::updateB(const Vector& grad_diff, Vector& memory, int time_step) {
    std::visit([&](auto& l) {
        if constexpr (HasUpdateB<decltype(l)>::value) {
            l.updateB(grad_diff, memory, time_step);
        }
    }, layer_);
}

const Matrix& AnyLayer::getWeights() const {
    return std::visit([](const auto& l) -> const Matrix& {
        static const Matrix dummy = Matrix::Zero(1, 1); // заглушка
        if constexpr (HasGetWeights<decltype(l)>::value) {
            return l.getWeights();
        } else {
            return dummy;
        }
    }, layer_);
}

const Vector& AnyLayer::getBiases() const {
    return std::visit([](const auto& l) -> const Vector& {
        static const Vector dummy = Vector::Zero(1); // заглушка
        if constexpr (HasGetBiases<decltype(l)>::value) {
            return l.getBiases();
        } else {
            return dummy;
        }
    }, layer_);
}

Index AnyLayer::getInputSize() const {
    return std::visit([](const auto& l) -> Index {
        if constexpr (HasGetInputSize<decltype(l)>::value) {
            return l.getInputSize();
        } else {
            return 0;
        }
    }, layer_);
}

Index AnyLayer::getOutputSize() const {
    return std::visit([](const auto& l) -> Index {
        if constexpr (HasGetOutputSize<decltype(l)>::value) {
            return l.getOutputSize();
        } else {
            return 0;
        }
    }, layer_);
}

} // namespace neural_network
