#pragma once

#include <utility>

#include "Math.h"

namespace neural_network
{
class DataLoader
{
public:
    enum NormalizeStatus : char
    {
        Active,
        NotActive
    };

    DataLoader(const Matrix& data, const Matrix& labels, Index batch_size,
               NormalizeStatus normalize_status = NotActive);

    class Iterator
    {
    public:
        Iterator(const Matrix& data, const Matrix& labels, Index batch_size, Index batch_index);

        bool operator!=(const Iterator& other) const;

        void operator++();

        std::pair<Matrix, Matrix> operator*() const;

    private:
        const Matrix& data_;
        const Matrix& labels_;
        Index batch_size_;
        Index batch_index_;
    };
    Iterator begin() const;
    Iterator end() const;

private:
    Matrix data_;
    Matrix labels_;
    Index batch_size_;
    Index num_batches_;

    Matrix normalization(Matrix data);
};

}  // namespace neural_network
