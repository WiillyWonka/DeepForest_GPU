#include "ScanUnit.cuh"

using std::vector;
using thrust::device_vector;

ScanUnit::ScanUnit(int n_estimators, int n_ferns, int depth, int win_size, int stride)
{
	random_ferns.reserve(n_estimators);
	for (int i = 0; i < n_estimators; i++)
		random_ferns.push_back(RandomFernsScan(win_size, stride, n_ferns, depth));
}

void ScanUnit::processBatch(thrust::device_vector<uint8_t>& data_batch, thrust::device_vector<uint32_t>& label_batch)
{
	for (auto& random_fern : random_ferns) 
		random_fern.processBatch(data_batch, label_batch);
}

vector<vector<float>> ScanUnit::transform(const vector<const vector<uint8_t>*>& data, uint32_t batch_size)
{
	/*
	thrust::device_vector<float> buffer (batch_size * n_classes * random_ferns.size());
	for (auto& random_fern : random_ferns) random_fern.tranformBatch(data_batch, label_batch);
	*/

	moveHost2Device();

	vector<vector<float>> out(data.size());
	device_vector<uint8_t> data_batch;
	vector<device_vector<float>> transformed(random_ferns.size());
	uint32_t current_size;
	for (int i = 0; i < data.size(); i += batch_size) {
		current_size = std::min(static_cast<int>(batch_size), static_cast<int>(data.size()) - i);
		data_batch = packBatch(data, current_size, i);

		for (int j = 0; j < transformed.size(); j++) random_ferns[j].transformBatch(data_batch, transformed[j], current_size);
		//for (int j = 0; j < buffer.size(); j++) out[i + j] = std::move(buffer[j]);
		cudaDeviceSynchronize();

		unpackTransformed(out, transformed, current_size, i);
		for (auto& vec : transformed) vec.clear();
	}

	releaseDevice();

	return out;
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
	const vector<const vector<uint8_t>*>& in,
	uint32_t batch_size,
	uint32_t start_idx)
{
	device_vector<uint8_t> out(batch_size * n_features);
	auto out_it = out.begin();
	auto data_it = in.begin() + start_idx;
	while (out_it != out.end())
	{
		thrust::copy((*data_it)->begin(), (*data_it)->end(), out_it);
		out_it += n_features;
		data_it++;
	}

	return out;
}

void ScanUnit::unpackTransformed(
	vector<vector<float>>& dst,
	vector<device_vector<float>>& src, 
	uint32_t batch_size,
	int index)
{
	int transformed_size = src[0].size() / batch_size;
	for (int i = 0; i < batch_size; i++) {
		dst[index + i] = vector<float>(transformed_size * src.size());
		for (int j = 0; j < src.size(); j++) {
			thrust::copy(src[j].begin() + transformed_size * i,
				src[j].begin() + transformed_size * (i + 1), dst[index + i].begin() + j * transformed_size);
		}
	}
}
