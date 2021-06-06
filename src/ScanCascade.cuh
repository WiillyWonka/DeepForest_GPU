#pragma once

#include <vector>
#include <thrust/device_vector.h>
#include <iostream>

#include "ScanUnit.cuh"
#include "json11/json11.hpp"
#include "utils.h"

using std::vector;

class ScanCascade
{
public:
	ScanCascade(int n_scan_units, int n_estimators, int n_ferns, int depth, int win_size, int stride);
	ScanCascade(const json11::Json::array& config);

	void fit(const vector<p_vector<uint8_t>>& X_train, const p_vector<uint32_t>& y_train, uint32_t batch_size);
	void calculateTransform(
		const vector<p_vector<uint8_t>>& data,
		uint32_t batch_size);
	void clearTransformed();
	const vector<p_vector<float>>& getTransformed(uint32_t index) const;
	uint32_t size() const { return scan_units.size(); };
	void setClassesNumber(uint32_t n_classes);
	void setFeaturesNumber(uint32_t n_features);

private:
	thrust::device_vector<uint8_t> packBatch(
		const vector<p_vector<uint8_t>>& in,
		uint32_t batch_size,
		uint32_t start_idx);
	thrust::device_vector<uint32_t> packBatch(
		const p_vector<uint32_t>& in, 
		uint32_t batch_size,
		uint32_t start_idx);

private:
	vector<ScanUnit> scan_units;
	uint32_t n_features;
};
