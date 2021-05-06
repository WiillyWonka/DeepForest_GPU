#pragma once

#include <vector>
#include <list>
#include <limits.h>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <cassert>

#include "ScanCascade.cuh"
#include "CascadeLevel.cuh"
#include "json11/json11.hpp"
#include "Timer.h"


using std::vector;

class DeepForest
{
public:
	DeepForest(const json11::Json& config);

	void fit(const std::vector<std::vector<uint8_t>>& X_train, const std::vector<uint32_t>& y_train,
		int img_height, int img_width, int batch_size);
	std::vector<uint32_t> predict(const std::vector<std::vector<uint8_t>>& X_test, int batch_size);

private:
	vector<uint32_t> predict(const vector<const vector<uint8_t>*>& X_test, int batch_size);

	vector<vector<float>> probaAveraging(vector<vector<float>> last_output);

	void getKFoldData(
		const vector<vector<uint8_t>>& in_X,
		const vector<uint32_t>& in_y,
		vector<const vector<uint8_t>*>& X_test,
		vector<uint32_t>& y_test,
		vector<const vector<uint8_t>*>& X_train,
		vector<uint32_t>& y_train);

	vector<vector<float>> getLastOutput(
		const vector<const vector<uint8_t>*>& data, uint32_t batch_size);
	vector<vector<float>> getLastTransformed(
		const vector<const vector<uint8_t>*>& data, uint32_t batch_size);
	uint32_t getClassNumber(const std::vector<uint32_t>& labels);
	double accuracy(std::vector<uint32_t>& test, std::vector<uint32_t>& pred);

private:
	ScanCascade scan_cascade;
	std::list<CascadeLevel> cascade;
	int n_classes, n_features, k = 5, n_random_ferns, n_ferns, depth;
	double tolerance = 0.001;
};

