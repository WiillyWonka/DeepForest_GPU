#pragma once

#include <vector>
#include <thrust/device_vector.h>

#include "RandomFernsScan.cuh"

using std::vector;

class ScanUnit
{
public:
	ScanUnit(int n_estimators = 2);
	
	void processBatch(thrust::device_vector<uint8_t>& data_batch, thrust::device_vector<uint32_t>& label_batch);
	std::vector<std::vector<float>> transform(const vector<const vector<uint8_t>*>& data, uint32_t batch_size);
	//const std::vector<float> getTranformed(int index) const;	
	void startFitting();
	void endFitting();
	void moveHost2Device();
	void releaseDevice();

	void setClassesNumber(uint32_t n_classes);
	void setFeaturesNumber(uint32_t n_features);

private:
	thrust::device_vector<uint8_t> packBatch(
		const vector<const vector<uint8_t>*>& in,
		uint32_t batch_size,
		uint32_t start_idx);

	void unpackTransformed(
		std::vector<std::vector<float>>& dst,
		std::vector<thrust::device_vector<float>>& src,
		uint32_t batch_size,
		int index);

private:
	std::vector<RandomFernsScan> random_ferns;
	std::vector<std::vector<float>> tranformed;
	uint32_t n_classes, n_features;
};