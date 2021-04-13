#include "CascadeLevel.cuh"

using std::vector;
using thrust::device_vector;

void CascadeLevel::fit(std::vector<std::vector<float>>& data, std::vector<uint32_t>& labels)
{
	startFitting();

	vector<vector<float>> out(data.size());
	device_vector<float> data_batch;
	vector<device_vector<float>> transformed(random_ferns.size());

	device_vector<float> data_batch;
	device_vector<uint32_t> label_batch;
	for (int i = 0; i < data.size(); i += batch_size) {
		data_batch = packBatch(data, i);
		label_batch = packBatch(labels, i);

		for (auto& random_fern : random_ferns) random_fern.processBatch(data_batch, label_batch);

		cudaDeviceSynchronize();
	}

	endFitting();
}

void CascadeLevel::predict(std::vector<std::vector<float>>& data)
{
	moveHost2Device();

	releaseDevice();
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
	uint32_t start_idx)
{
	device_vector<uint32_t> out(batch_size);
	thrust::copy(in.begin() + start_idx, in.begin() + start_idx + batch_size, out.begin());
	return out;
}

device_vector<float> CascadeLevel::packBatch(
	const vector<vector<float>>& in,
	uint32_t start_idx)
{
	device_vector<uint8_t> out(batch_size);
	auto out_it = out.begin();
	auto data_it = in.begin() + start_idx;
	uint32_t sample_size = data_it->size();
	while (out_it != out.end())
	{
		thrust::copy(data_it->begin(), data_it->end(), out_it);
		out_it += sample_size;
		data_it++;
	}
}