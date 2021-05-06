#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <algorithm>

#include "Fern.cuh"


class RandomFerns
{
public:
	RandomFerns(int n_classes, int n_features, int n_estimators=3, int depth=5);
	void processBatch(thrust::device_vector<float>& data, thrust::device_vector<uint32_t>& labels);
	std::vector<std::vector<float>> transformBatch(thrust::device_vector<float>& data, uint32_t batch_size);
	std::vector<float> predictProbaSingle(std::vector<double>& X_test);

	void startFitting();
	void endFitting();

	void moveHost2Device();
	void releaseDevice();

private:
	std::vector<std::vector<float>> unpackProba(thrust::device_vector<float>& proba);

private:
	int n_classes, n_features;
	std::vector<Fern> ferns;
};

