#pragma once

#include <vector>
#include <thrust/device_vector.h>

#include "ScanUnit.cuh"

class ScanCascade
{
public:
	ScanCascade();

	void fit(const std::vector<std::vector<uint8_t>>& X_train, const std::vector<uint32_t>& y_train);
	std::vector<std::vector<float>> transform(
		const std::vector<std::vector<uint8_t>&>& data,
		uint32_t index);
	uint32_t size() const { return scan_units.size(); };
	void setClassesNumber(uint32_t n_classes);
	void setFeaturesNumber(uint32_t n_features);

private:
	thrust::device_vector<uint8_t> packBatch(
		const std::vector<std::vector<uint8_t>>& in,
		uint32_t start_idx);
	thrust::device_vector<uint32_t> packBatch(
		const std::vector<uint32_t>& in,
		uint32_t start_idx);

private:
	std::vector<ScanUnit> scan_units;
	uint32_t batch_size = 100, n_features;
};
