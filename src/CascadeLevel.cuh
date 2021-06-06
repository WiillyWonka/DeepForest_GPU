#pragma once

#include <vector>

#include "RandomFerns.cuh"
#include "utils.h"

using std::vector;

class CascadeLevel
{
public:
	CascadeLevel(int n_estimators, int n_ferns, int depth, int n_classes, int n_features);
	void fit(const vector<const p_vector<float>*>& data,
		const p_vector<uint32_t>& label, uint32_t batch_size);
	void calculateTransform(const vector<p_vector<float>>& data, uint32_t batch_size);
	const vector<p_vector<float>>& getTransfomed() const;
	void clearTranformed();
	size_t size() const { return random_ferns.size(); };

	void moveHost2Device();
	void releaseDevice();

	void startFitting();
	void endFitting();

private:
	thrust::device_vector<float> packBatch(
		const vector<const p_vector<float>*>& in,
		uint32_t start_idx,
		uint32_t batch_size);
	thrust::device_vector<float> packBatch(
		const vector<p_vector<float>>& in,
		uint32_t start_idx,
		uint32_t batch_size);
	thrust::device_vector<uint32_t> packBatch(
		const p_vector<uint32_t>& in,
		uint32_t start_idx,
		uint32_t batch_size);

	void unpackTransformed(vector<p_vector<float>>& transformed,
		vector<vector<vector<float>>> buffer, int idx, int batch_size);

private:
	vector<p_vector<float>> transformed;
	vector<RandomFerns> random_ferns;
	uint32_t n_classes;
};