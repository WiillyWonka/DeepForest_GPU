#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <ctime>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "utils.h"

class Fern
{
public:
	Fern(int n_classes, int n_features, int depth=10);

	void moveHost2Device();
	void releaseDevice();
	void startFitting();
	void endFitting();
	void processBatch(thrust::device_vector<float>& data, thrust::device_vector<uint32_t>& labels);
	void transformBatch(
		thrust::device_vector<float>& X_test,
		thrust::device_vector<float>& proba,
		uint32_t batch_size);
	std::vector<float> predictProbaSingle(std::vector<double>& X_test);

private:
	void normalizeHist();

private:
	int depth, n_classes, n_features;
	double min_feature = 0, max_feature = 1;
	p_vector<int> h_feature_idx;
	thrust::device_vector<int> d_feature_idx;
	p_vector<float>	h_thresholds, h_hist;
	thrust::device_vector<float> d_thresholds, d_hist;
	cudaStream_t stream;
};