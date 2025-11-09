#include "AnyLayer.h"

namespace neural_network
{

template <typename, typename = std::void_t<>>
struct has_getWeights : std::false_type {
};

template <typename T>
struct has_getWeights<T, std::void_t<decltype(std::declval<T>().getWeights())>> : std::true_type {
};

template <typename, typename = std::void_t<>>
struct has_getBiases : std::false_type {
};

template <typename T>
struct has_getBiases<T, std::void_t<decltype(std::declval<T>().getBiases())>> : std::true_type {
};

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
    return std::visit(
        [&](const auto& l) -> Matrix {
            using LayerT = std::decay_t<decltype(l)>;
            auto ptr = static_cast<Matrix (LayerT::*)(const Matrix&) const>(&LayerT::evaluate);
            return (l.*ptr)(input);
        },
        layer_);
}

Matrix AnyLayer::getBackpropError(const Matrix& a, const Matrix& z, const Matrix& b) const
{
    return std::visit(
        [&](const auto& l) -> Matrix {
            if constexpr (std::is_invocable_v<
                              decltype(&std::decay_t<decltype(l)>::getBackpropError),
                              std::decay_t<decltype(l)>, Matrix, Matrix, Matrix>) {
                return l.getBackpropError(a, z, b);
            } else {
                return a;
            }
        },
        layer_);
}

Matrix AnyLayer::getGradW(const Matrix& a, const Matrix& z, const Matrix& b) const
{
    return std::visit(
        [&](const auto& l) -> Matrix {
            if constexpr (std::is_same<std::decay_t<decltype(l)>, DropoutLayer>::value) {
                return Matrix();
            } else if constexpr (std::is_invocable_v<decltype(&std::decay_t<decltype(l)>::getGradW),
                                                     std::decay_t<decltype(l)>, Matrix, Matrix,
                                                     Matrix>) {
                return l.getGradW(a, z, b);
            } else {
                return Matrix();
            }
        },
        layer_);
}

Matrix AnyLayer::getGradB(const Matrix& a, const Matrix& z, const Matrix& b) const
{
    return std::visit(
        [&](const auto& l) -> Matrix {
            if constexpr (std::is_same<std::decay_t<decltype(l)>, DropoutLayer>::value) {
                return Matrix();
            } else if constexpr (std::is_invocable_v<decltype(&std::decay_t<decltype(l)>::getGradB),
                                                     std::decay_t<decltype(l)>, Matrix, Matrix,
                                                     Matrix>) {
                return l.getGradB(a, z, b);
            } else {
                return Matrix();
            }
        },
        layer_);
}

bool AnyLayer::hasWeights() const
{
    return std::visit(
        [](const auto& l) -> bool {
            if constexpr (std::is_same_v<std::decay_t<decltype(l)>, DropoutLayer>) {
                return false;
            } else if constexpr (has_getWeights<std::decay_t<decltype(l)>>::value) {
                return true;
            } else {
                return false;
            }
        },
        layer_);
}

void AnyLayer::updateW(const Matrix& grad_diff, Matrix& memory, int time_step)
{
    std::visit(
        [&](auto& l) {
            if constexpr (std::is_invocable_v<decltype(&std::decay_t<decltype(l)>::updateW),
                                              std::decay_t<decltype(l)>, Matrix, Matrix&, int>) {
                l.updateW(grad_diff, memory, time_step);
            }
        },
        layer_);
}

void AnyLayer::updateB(const Vector& grad_diff, Vector& memory, int time_step)
{
    std::visit(
        [&](auto& l) {
            if constexpr (std::is_invocable_v<decltype(&std::decay_t<decltype(l)>::updateB),
                                              std::decay_t<decltype(l)>, Vector, Vector&, int>) {
                l.updateB(grad_diff, memory, time_step);
            }
        },
        layer_);
}

const Matrix& AnyLayer::getWeights() const
{
    return std::visit(
        [](const auto& l) -> const Matrix& {
            static const Matrix dummy = Matrix::Zero(0, 0);
            if constexpr (has_getWeights<std::decay_t<decltype(l)>>::value) {
                return l.getWeights();
            } else {
                return dummy;
            }
        },
        layer_);
}

const Vector& AnyLayer::getBiases() const
{
    return std::visit(
        [](const auto& l) -> const Vector& {
            static const Vector dummy = Vector::Zero(0);
            if constexpr (has_getBiases<std::decay_t<decltype(l)>>::value) {
                return l.getBiases();
            } else {
                return dummy;
            }
        },
        layer_);
}

Index AnyLayer::getInputSize() const
{
    return std::visit(
        [](const auto& l) -> Index {
            if constexpr (std::is_invocable_v<decltype(&std::decay_t<decltype(l)>::getInputSize),
                                              std::decay_t<decltype(l)>>) {
                return l.getInputSize();
            } else {
                return 0;
            }
        },
        layer_);
}

Index AnyLayer::getOutputSize() const
{
    return std::visit(
        [](const auto& l) -> Index {
            if constexpr (std::is_invocable_v<decltype(&std::decay_t<decltype(l)>::getOutputSize),
                                              std::decay_t<decltype(l)>>) {
                return l.getOutputSize();
            } else {
                return 0;
            }
        },
        layer_);
}

}  // namespace neural_network
