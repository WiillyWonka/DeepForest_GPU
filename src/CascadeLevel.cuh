#pragma once

#include <vector>

#include "RandomFerns.cuh"

class CascadeLevel
{
public:
	CascadeLevel(uint32_t n_estimators, uint32_t batch_size);
	void fit(std::vector<std::vector<float>>& data, std::vector<uint32_t>& label);
	void predict(std::vector<std::vector<float>>& data);
	size_t size() { return random_ferns.size(); };

	void moveHost2Device();
	void releaseDevice();

	void startFitting();
	void endFitting();

private:
	thrust::device_vector<float> packBatch(
		const std::vector<std::vector<float>>& in,
		uint32_t start_idx);
	thrust::device_vector<uint32_t> packBatch(
		const std::vector<uint32_t>& in,
		uint32_t start_idx);

private:
	std::vector<RandomFerns> random_ferns;
	uint32_t batch_size;
};