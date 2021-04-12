#pragma once

#include <vector>
#include <thrust/device_vector.h>

#include "RandomFernsScan.cuh"

class ScanUnit
{
public:
	ScanUnit(uint32_t batch_size);
	
	void processBatch(thrust::device_vector<uint8_t>& data_batch, thrust::device_vector<uint32_t>& label_batch);
	std::vector<std::vector<float>> transform(const std::vector<std::vector<uint8_t>&>& data);
	//const std::vector<float> getTranformed(int index) const;	
	void startFitting();
	void endFitting();
	void moveHost2Device();
	void releaseDevice();
	void setClassesNumber(uint32_t n_classes);
	void setFeaturesNumber(uint32_t n_features);

private:
	thrust::device_vector<uint8_t> packBatch(
		const std::vector<std::vector<uint8_t>&>& in,
		uint32_t start_idx);

	void unpackTransformed(vector<vector<float>>& dst, vector<thrust::device_vector<float>>& src, int index);

private:
	std::vector<RandomFernsScan> random_ferns;
	std::vector<std::vector<float>> tranformed;
	uint32_t n_classes, n_features, batch_size;
};