#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <cassert>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <algorithm>

#include "Fern.cuh"


class RandomFerns
{
public:
	RandomFerns(int n_classes, int n_features, int n_estimators=50, int depth=10);
	void fit(std::vector<double>& X_train,
		std::vector<int>& Y_train,
		int batch_size=30);
	std::vector<double> predictProba(std::vector<double>& X_test);

private:
	int n_classes, n_features;
	std::vector<Fern> ferns;
};

