#include "ScanUnit.cuh"

using std::vector;
using thrust::device_vector;

void ScanUnit::processBatch(thrust::device_vector<uint8_t>& data_batch, thrust::device_vector<uint32_t>& label_batch)
{
	for (auto& random_fern : random_ferns) random_fern.processBatch(data_batch, label_batch);
}

vector<vector<float>> ScanUnit::transform(const vector<vector<uint8_t>&>& data)
{
	/*
	thrust::device_vector<float> buffer (batch_size * n_classes * random_ferns.size());
	for (auto& random_fern : random_ferns) random_fern.tranformBatch(data_batch, label_batch);
	*/

	moveHost2Device();

	vector<vector<float>> out(data.size());
	device_vector<uint8_t> data_batch;
	vector<device_vector<float>> transformed(random_ferns.size());
	
	for (int i = 0; i < data.size(); i += batch_size) {
		data_batch = packBatch(data, i);

		for (int j = 0; j < transformed.size(); j++) random_ferns[j].tranformBatch(data_batch, transformed[j]);
		//for (int j = 0; j < buffer.size(); j++) out[i + j] = std::move(buffer[j]);
		cudaDeviceSynchronize();

		unpackTransformed(out, transformed, i);
	}

	releaseDevice();
}

void ScanUnit::startFitting()
{
	for (auto& random_fern : random_ferns) random_fern.startFitting();
}

void ScanUnit::endFitting()
{
	for (auto& random_fern : random_ferns) random_fern.endFitting();
}

void ScanUnit::moveHost2Device()
{
	for (auto& random_fern : random_ferns) random_fern.moveHost2Device();
}

void ScanUnit::releaseDevice()
{
	for (auto& random_fern : random_ferns) random_fern.releaseDevice();
}

void ScanUnit::setClassesNumber(uint32_t n_classes)
{
	this->n_classes = n_classes;
	for (auto& random_fern : random_ferns) random_fern.setClassesNumber(n_classes);
}

void ScanUnit::setFeaturesNumber(uint32_t n_features)
{
	this->n_features = n_features;
	for (auto& random_fern : random_ferns) random_fern.setFeaturesNumber(n_features);
}

device_vector<uint8_t> ScanUnit::packBatch(
	const std::vector<std::vector<uint8_t>&>& in,
	uint32_t start_idx)
{
	device_vector<uint8_t> out(batch_size * n_features);
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

void ScanUnit::unpackTransformed(vector<vector<float>>& dst, vector<thrust::device_vector<float>>& src, int index)
{
	for (int i = 0; i < src.size(); i++) {
		thrust::copy(src[i].begin(), src[i].end(), dst[index].begin())
	}
}
