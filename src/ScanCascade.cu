﻿#include "ScanCascade.cuh"

using std::vector;
using thrust::device_vector;

ScanCascade::ScanCascade() 
	: scan_units(2)
{}


void ScanCascade::fit(const vector<vector<uint8_t>>& data, const vector<uint32_t>& labels)
{
	for (auto& unit : scan_units) unit.startFitting();

	device_vector<uint8_t> data_batch;
	device_vector<uint32_t> label_batch;
	for (int i = 0; i < data.size(); i+=batch_size) {
		data_batch = packBatch(data, i);
		label_batch = packBatch(labels, i);

		for (auto& unit : scan_units) unit.processBatch(data_batch, label_batch);

		cudaDeviceSynchronize();
	}

	for (auto& unit : scan_units) unit.endFitting();
}

vector<vector<float>> ScanCascade::transform(const vector<vector<uint8_t>&>& data, uint32_t index)
{
	return scan_units[index].transform(data);

	/*
	scan_units[index].moveHost2Device();

	vector<vector<float>> out(data.size()), buffer;
	device_vector<uint8_t> data_batch;
	for (int i = 0; i < data.size(); i += batch_size) {
		data_batch = packBatch(data, i);

		buffer = scan_units[index].transformBatch(data_batch);
		for (int j = 0; j < buffer.size(); j++) out[i + j] = std::move(buffer[j]);

		cudaDeviceSynchronize();
	}

	scan_units[index].moveHost2Device();
	*/
}



void ScanCascade::setClassesNumber(uint32_t n_classes)
{
	for (auto& unit : scan_units) unit.setClassesNumber(n_classes);
}

void ScanCascade::setFeaturesNumber(uint32_t n_features)
{
	this->n_features = n_features;
	for (auto& unit : scan_units) unit.setFeaturesNumber(n_features);
}


device_vector<uint8_t> ScanCascade::packBatch(
	const vector<vector<uint8_t>>& in,
	uint32_t start_idx)
{
	device_vector<uint8_t> out(batch_size);
	thrust::copy(in.begin() + start_idx, in.begin() + start_idx + batch_size, out.begin());
	return out;
}

device_vector<uint8_t> ScanCascade::packBatch(
	const vector<vector<uint8_t>>& in,
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