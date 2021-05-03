#include "CascadeLevel.cuh"

using std::vector;
using thrust::device_vector;

CascadeLevel::CascadeLevel(uint32_t n_estimators, uint32_t n_classes, uint32_t n_features)
	: random_ferns(n_estimators, RandomFerns(n_classes, n_features, 5, 3))
	, n_classes(n_classes)
{
}

void CascadeLevel::fit(
	const std::vector<std::vector<float>>& data,
	const std::vector<uint32_t>& labels,
	uint32_t batch_size)
{
	startFitting();

	vector<vector<float>> out(data.size());
	device_vector<float> data_batch;
	device_vector<uint32_t> label_batch;
	
	for (int i = 0; i < data.size(); i += batch_size) {
		data_batch = packBatch(data, i, batch_size);
		label_batch = packBatch(labels, i, batch_size);

		for (auto& random_fern : random_ferns) random_fern.processBatch(data_batch, label_batch);

		cudaDeviceSynchronize();
	}

	endFitting();
}

vector<vector<float>> CascadeLevel::transform(vector<vector<float>>& data, uint32_t batch_size)
{
	moveHost2Device();

	device_vector<float> data_batch;
	vector<vector<float>> transformed(data.size());
	vector<vector<vector<float>>> buffer(random_ferns.size());

	for (int i = 0; i < data.size(); i += batch_size) {
		data_batch = packBatch(data, i, batch_size);

		for (int j = 0; j < random_ferns.size(); j++) 
			buffer[j] = random_ferns[j].transformBatch(data_batch, batch_size);

		cudaDeviceSynchronize();

		unpackTransformed(transformed, buffer, i, batch_size);
	}

	releaseDevice();
	return transformed;
}

void CascadeLevel::moveHost2Device()
{
	for (auto& random_fern : random_ferns) random_fern.moveHost2Device();
}

void CascadeLevel::releaseDevice()
{
	for (auto& random_fern : random_ferns) random_fern.releaseDevice();
}

void CascadeLevel::startFitting()
{
	for (auto& random_fern : random_ferns) random_fern.startFitting();
}

void CascadeLevel::endFitting()
{
	for (auto& random_fern : random_ferns) random_fern.endFitting();
}

device_vector<uint32_t> CascadeLevel::packBatch(
	const vector<uint32_t>& in,
	uint32_t start_idx,
	uint32_t batch_size)
{
	device_vector<uint32_t> out(batch_size);
	thrust::copy(in.begin() + start_idx, in.begin() + start_idx + batch_size, out.begin());
	return out;
}

void CascadeLevel::unpackTransformed(vector<vector<float>>& transformed,
	vector<vector<vector<float>>> buffer, int start_idx, int batch_size)
{
	for (int i = 0; i < batch_size; i++) {
		transformed[i + start_idx] = vector<float>(n_classes * random_ferns.size());
		auto transformed_it = transformed[i + start_idx].begin();
		for (auto& proba : buffer) {
			std::copy(proba[i].begin(), proba[i].end(), transformed_it);
			transformed_it += proba.size();
		}
	}
}

device_vector<float> CascadeLevel::packBatch(
	const vector<vector<float>>& in,
	uint32_t start_idx,
	uint32_t batch_size)
{
	device_vector<float> out(batch_size * in[0].size());
	auto out_it = out.begin();
	auto data_it = in.begin() + start_idx;
	uint32_t sample_size = data_it->size();
	while (out_it != out.end())
	{
		thrust::copy(data_it->begin(), data_it->end(), out_it);
		out_it += sample_size;
		data_it++;
	}

	return out;
}