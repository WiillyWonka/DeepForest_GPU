#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <ctime>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <iostream>

#include "utils.h"

class FernScan
{
public:
	FernScan(int win_size, int stride, int depth = 3);

	void processBatch(thrust::device_vector<uint8_t>& data, thrust::device_vector<uint32_t>& labels);
	void transformBatch(
		thrust::device_vector<uint8_t>& data,
		thrust::device_vector<float>& transformed,
		uint32_t batch_size);

	void moveHost2Device();
	void releaseDevice();
	void startFitting();
	void endFitting();

	void setClassesNumber(uint32_t n_classes);
	void setFeaturesNumber(uint32_t n_features);

private:
	void normalizeHist();
private:
	uint32_t depth, n_classes, n_features, win_size, stride;
	uint32_t image_width = 28, image_height = 28;
	uint8_t min_feature = 0, max_feature = 255;
	p_vector<int> h_feature_idx, h_thresholds;
	p_vector<float> h_hist;
	thrust::device_vector<int> d_feature_idx, d_thresholds;
	thrust::device_vector<float> d_hist;
	cudaStream_t stream;
};