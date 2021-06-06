#pragma once

#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

template <typename T>
using p_vector = thrust::host_vector<T, thrust::system::cuda::experimental::pinned_allocator<T>>;
