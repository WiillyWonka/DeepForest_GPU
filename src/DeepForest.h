#pragma once

#include <vector>
#include <list>
#include <limits.h>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <cassert>

#include "cuda_profiler_api.h"

#include "ScanCascade.cuh"
#include "CascadeLevel.cuh"
#include "json11/json11.hpp"
#include "Timer.h"
#include "Logger.hpp"
#include "utils.h"


using std::vector;

class DeepForest
{
public:
	DeepForest(const json11::Json& config);

	void fit(const vector<p_vector<uint8_t>>& X_train, const p_vector<uint32_t>& y_train,
		int img_height, int img_width, int batch_size);
	std::vector<uint32_t> predict(const std::vector<p_vector<uint8_t>>& X_test, int batch_size);

private:
	vector<uint32_t> predict(const vector<const p_vector<uint8_t>*>& X_test, int batch_size);

	vector<vector<float>> probaAveraging(const vector<p_vector<float>>& last_output);
	vector<vector<float>> probaAveraging(const vector<const p_vector<float>*>& last_output);

	void getKFoldIndices(
		vector<uint32_t>& train_indices,
		vector<uint32_t>& test_indices,
		size_t dataset_size);

	void getSubsetByIndices(
		const vector<p_vector<float>>& X_in,
		const p_vector<uint32_t>& y_in,
		const vector<uint32_t>& indices,
		vector<const p_vector<float>*>& X_out,
		p_vector<uint32_t>& y_out);

	vector<p_vector<float>> getLastTransformed();
	vector<p_vector<float>> concatenate(const vector<p_vector<float>>& first, const vector<p_vector<float>> second);

	uint32_t getClassNumber(const p_vector<uint32_t>& labels);

	double accuracy(p_vector<uint32_t>& label, vector<vector<float>>& proba);

private:
	ScanCascade scan_cascade;
	std::list<CascadeLevel> cascades;
	int n_classes, n_features, k = 5, n_random_ferns, n_ferns, depth, cascade_size;
	double tolerance = 0.001;
};

