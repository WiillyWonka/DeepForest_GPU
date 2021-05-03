﻿#pragma once

#include <vector>

#include "RandomFerns.cuh"

class CascadeLevel
{
public:
	CascadeLevel(uint32_t n_estimators, uint32_t n_classes, uint32_t n_features);
	void fit(const std::vector<std::vector<float>>& data, const std::vector<uint32_t>& label, uint32_t batch_size);
	std::vector<std::vector<float>> transform(std::vector<std::vector<float>>& data, uint32_t batch_size);
	size_t size() { return random_ferns.size(); };

	void moveHost2Device();
	void releaseDevice();

	void startFitting();
	void endFitting();

private:
	thrust::device_vector<float> packBatch(
		const std::vector<std::vector<float>>& in,
		uint32_t start_idx,
		uint32_t batch_size);
	thrust::device_vector<uint32_t> packBatch(
		const std::vector<uint32_t>& in,
		uint32_t start_idx,
		uint32_t batch_size);

	void unpackTransformed(std::vector<std::vector<float>>& transformed,
		std::vector<std::vector<std::vector<float>>> buffer, int idx, int batch_size);

private:
	std::vector<RandomFerns> random_ferns;
	uint32_t n_classes;
};