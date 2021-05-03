#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <ctime>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <iostream>

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
	uint32_t image_width = 28, image_height = 28; //TODO: remove hardcode
	uint8_t min_feature = 0, max_feature = 255;
	thrust::host_vector<uint32_t> h_feature_idx;
	thrust::host_vector<uint8_t> h_thresholds;
	thrust::host_vector<float> h_hist;
	thrust::device_vector<uint32_t> d_feature_idx;
	thrust::device_vector<uint8_t> d_thresholds;
	thrust::device_vector<float> d_hist;
};