#include "CascadeLevel.cuh"

using std::vector;
using thrust::device_vector;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

CascadeLevel::CascadeLevel(int n_estimators, int n_ferns, int depth, int n_classes, int n_features)
	: random_ferns(n_estimators, RandomFerns(n_classes, n_features, n_ferns, depth))
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
	uint32_t current_size;
	for (int i = 0; i < data.size(); i += batch_size) {
		current_size = std::min(static_cast<int>(batch_size), static_cast<int>(data.size()) - i);
		data_batch = packBatch(data, i, current_size);
		label_batch = packBatch(labels, i, current_size);

		gpuErrchk(cudaPeekAtLastError());
		for (auto& random_fern : random_ferns) random_fern.processBatch(data_batch, label_batch);

		gpuErrchk(cudaDeviceSynchronize());
		data_batch = device_vector<float>();
		label_batch = device_vector<uint32_t>();
	}

	endFitting();
}

vector<vector<float>> CascadeLevel::transform(vector<vector<float>>& data, uint32_t batch_size)
{
	moveHost2Device();

	device_vector<float> data_batch;
	vector<vector<float>> transformed(data.size());
	vector<vector<vector<float>>> buffer(random_ferns.size());

	uint32_t current_size;
	for (int i = 0; i < data.size(); i += batch_size) {
		current_size = std::min(static_cast<int>(batch_size), static_cast<int>(data.size()) - i - 1);
		data_batch = packBatch(data, i, current_size);

		for (int j = 0; j < random_ferns.size(); j++) 
			buffer[j] = random_ferns[j].transformBatch(data_batch, current_size);

		cudaDeviceSynchronize();

		unpackTransformed(transformed, buffer, i, current_size);
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
			transformed_it += proba[i].size();
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