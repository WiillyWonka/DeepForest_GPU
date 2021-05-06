#include "ScanCascade.cuh"

using std::vector;
using thrust::device_vector;

ScanCascade::ScanCascade(
	int n_scan_units, int n_estimators, int n_ferns, int depth, int win_size, int stride)
	: scan_units(n_scan_units, ScanUnit(n_estimators, n_ferns, depth, win_size, stride))
{}


void ScanCascade::fit(const vector<vector<uint8_t>>& data, const vector<uint32_t>& labels, uint32_t batch_size)
{
	for (auto& unit : scan_units) unit.startFitting();

	device_vector<uint8_t> data_batch;
	device_vector<uint32_t> label_batch;
	uint32_t current_size;
	for (int i = 0; i < data.size(); i += batch_size) {
		current_size = std::min(static_cast<int>(batch_size), static_cast<int>(data.size()) - i);
		data_batch = packBatch(data, current_size, i);
		label_batch = packBatch(labels, current_size, i);

		/*
		for (auto i : data_batch)
			std::cout << (int)i << std::endl;
		*/
		for (auto& unit : scan_units) 
			unit.processBatch(data_batch, label_batch);

		cudaDeviceSynchronize();
		data_batch = device_vector<uint8_t>();
		label_batch = device_vector<uint32_t>();
	}

	for (auto& unit : scan_units) unit.endFitting();
}

vector<vector<float>> ScanCascade::transform(const vector<const vector<uint8_t>*>& data,
	uint32_t index, uint32_t batch_size)
{
	return scan_units[index].transform(data, batch_size);
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


device_vector<uint32_t> ScanCascade::packBatch(
	const vector<uint32_t>& in,
	uint32_t batch_size,
	uint32_t start_idx)
{
	device_vector<uint32_t> out(batch_size);
	thrust::copy(in.begin() + start_idx, in.begin() + start_idx + batch_size, out.begin());
	return out;
}

device_vector<uint8_t> ScanCascade::packBatch(
	const vector<vector<uint8_t>>& in,
	uint32_t batch_size,
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

	return out;
}