#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <ctime>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class Fern
{
public:
	Fern();
	Fern(int n_classes, int n_features, int depth=10);

	void moveHost2Device();
	void moveDevice2Host();
	void startFitting();
	void endFitting();
	void processBatch(thrust::device_vector<double>& X, thrust::device_vector<int>& Y, int batch_size);
	std::vector<double> predictProba(std::vector<double>& X_train);

private:
	void normalizeHist();

private:
	int depth, n_classes, n_features;
	double min_feature = 0, max_feature = 1;
	thrust::host_vector<int> h_feature_idx, h_hist;
	thrust::device_vector<int> d_feature_idx, d_hist;
	thrust::host_vector<double> h_thresholds;
	thrust::device_vector<double> d_thresholds;
};