#include "DataLoader.h"
#include <cassert>

namespace neural_network
{
DataLoader::DataLoader(const Matrix& data, const Matrix& labels, Index batch_size) 
        : data_(data), labels_(labels), batch_size_(batch_size),
         num_batches_((data.rows() + batch_size - 1) / batch_size) 
{    
}

DataLoader::Iterator DataLoader::begin() const
{
    return Iterator(data_, labels_, batch_size_, 0);
}

DataLoader::Iterator DataLoader::end() const
{
    return Iterator(data_, labels_, batch_size_, num_batches_);
}

DataLoader::Iterator::Iterator(const Matrix& data, const Matrix& labels, Index batch_size, Index batch_index)
    : data_(data), labels_(labels), batch_size_(batch_size), batch_index_(batch_index) {}

bool DataLoader::Iterator::operator!=(const Iterator& other) const
{
    return batch_index_ != other.batch_index_;
}

void DataLoader::Iterator::operator++() { ++batch_index_; }

std::pair<Matrix, Matrix> DataLoader::Iterator::operator*() const 
{
    Index start_idx = batch_size_ * batch_index_;
    Index end_idx = std::min(start_idx + batch_size_, static_cast<Index>(data_.rows()));

    return {data_.block(start_idx, 0, end_idx - start_idx, data_.cols()),
            labels_.block(start_idx, 0, end_idx - start_idx, labels_.cols())};
}

} // neural_network
