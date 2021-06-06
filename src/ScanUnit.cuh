#pragma once

#include <vector>
#include <thrust/device_vector.h>

#include "RandomFernsScan.cuh"
#include "utils.h"

using std::vector;

class ScanUnit
{
public:
	ScanUnit(int n_estimators, int n_ferns, int depth, int win_size, int stride);
	
	void processBatch(thrust::device_vector<uint8_t>& data_batch, thrust::device_vector<uint32_t>& label_batch);
	void calculateTransform(const vector<p_vector<uint8_t>>& data, uint32_t batch_size);
	void clearTransformed();
	const vector<p_vector<float>>& getTranformed() const;	
	void startFitting();
	void endFitting();
	void moveHost2Device();
	void releaseDevice();

	void setClassesNumber(uint32_t n_classes);
	void setFeaturesNumber(uint32_t n_features);

private:
	thrust::device_vector<uint8_t> packBatch(
		const vector<p_vector<uint8_t>>& in,
		uint32_t batch_size,
		uint32_t start_idx);

	void unpackTransformed(
		vector<p_vector<float>>& dst,
		std::vector<thrust::device_vector<float>>& src,
		uint32_t batch_size,
		int index);

private:
	vector<RandomFernsScan> random_ferns;
	vector<p_vector<float>> transformed;
	uint32_t n_classes, n_features;
};