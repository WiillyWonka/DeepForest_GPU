#pragma once

#include <vector>
#include <list>
#include <limits.h>
#include <numeric>
#include <algorithm>

#include "ScanCascade.cuh"
#include "CascadeLevel.cuh"

class DeepForest
{
public:
	DeepForest();

	void fit(const std::vector<std::vector<uint8_t>>& X_train, const std::vector<uint32_t>& y_train);
	std::vector<uint32_t> predict(const std::vector<std::vector<uint8_t>>& X_test);
	std::vector<uint32_t> predict(const std::vector<std::vector<uint8_t>&>& X_test);

private:
	void getKFoldData(
		const std::vector<std::vector<uint8_t>>& in_X,
		const std::vector<uint32_t>& in_y,
		std::vector<std::vector<uint8_t>&>& X_test,
		std::vector<uint32_t>& y_test,
		std::vector<std::vector<uint8_t>&>& X_train,
		std::vector<uint32_t>& y_train);

	std::vector<std::vector<float>> getLastTransformed(const std::vector<std::vector<uint8_t>&>& data);
	uint32_t getClassNumber(const std::vector<uint32_t>& labels);
	double accuracy(std::vector<uint32_t>& test, std::vector<uint32_t>& pred);

private:
	ScanCascade scan_cascade;
	std::list<CascadeLevel> cascade;
	uint32_t n_classes, n_features, k = 5;
	double tolerance = 0.1;
};

