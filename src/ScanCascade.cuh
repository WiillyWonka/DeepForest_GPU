#pragma once

#include <vector>
#include <thrust/device_vector.h>
#include <iostream>

#include "ScanUnit.cuh"

class ScanCascade
{
public:
	ScanCascade(int n_scan_units, int n_estimators, int n_ferns, int depth, int win_size, int stride);

	void fit(const std::vector<std::vector<uint8_t>>& X_train, const std::vector<uint32_t>& y_train, uint32_t batch_size);
	void calculateTransform(
		const vector<vector<uint8_t>>& data,
		uint32_t batch_size);
	void clearTransformed();
	const std::vector<std::vector<float>>& getTransformed(uint32_t index) const;
	uint32_t size() const { return scan_units.size(); };
	void setClassesNumber(uint32_t n_classes);
	void setFeaturesNumber(uint32_t n_features);

private:
	thrust::device_vector<uint8_t> packBatch(
		const std::vector<std::vector<uint8_t>>& in,
		uint32_t batch_size,
		uint32_t start_idx);
	thrust::device_vector<uint32_t> packBatch(
		const std::vector<uint32_t>& in, 
		uint32_t batch_size,
		uint32_t start_idx);

private:
	std::vector<ScanUnit> scan_units;
	uint32_t n_features;
};
