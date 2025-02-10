#pragma once

#include "Math.h"
#include <utility>

namespace neural_network
{
class DataLoader {
public:
    DataLoader(const Matrix& data, const Matrix& labels, Index batch_size);

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
    const Matrix& data_;
    const Matrix& labels_;
    Index batch_size_;
    Index num_batches_;
};

} // namespace neural_network
