#pragma once

#include <vector>
#include <list>
#include <limits.h>
#include <numeric>
#include <algorithm>
#include <iostream>

#include "ScanCascade.cuh"
#include "CascadeLevel.cuh"


using std::vector;

class DeepForest
{
public:
	DeepForest();

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
	uint32_t n_classes, n_features, k = 3, n_random_ferns = 2;
	double tolerance = 0.1;
};

