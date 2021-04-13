#include "RandomFernsScan.cuh"

using std::vector;
using thrust::device_vector;

void RandomFernsScan::setClassesNumber(uint32_t n_classes)
{
	this->n_classes = n_classes;
	for (auto& fern : ferns) fern.setClassesNumber(n_classes);
}

void RandomFernsScan::setFeaturesNumber(uint32_t n_features)
{
	this->n_features = n_features;
	for (auto& fern : ferns) fern.setFeaturesNumber(n_features);
}

void RandomFernsScan::startFitting()
{
	for (auto& fern : ferns) fern.startFitting();
}

void RandomFernsScan::endFitting()
{
	for (auto& fern : ferns) fern.endFitting();
}

void RandomFernsScan::moveHost2Device()
{
	for (auto& fern : ferns) fern.moveHost2Device();
}

void RandomFernsScan::releaseDevice()
{
	for (auto& fern : ferns) fern.releaseDevice();
}

void RandomFernsScan::processBatch(device_vector<uint8_t>& data, device_vector<uint32_t>& labels)
{
	for (auto& fern : ferns) fern.processBatch(data, labels);
}

void normalizeTransform(float* tranformed, int n_windows, int n_classes)
{
	int sample_idx = threadIdx.x;
	int win_idx = threadIdx.y;
	int begin_idx = sample_idx * n_windows * n_classes + win_idx * n_classes;

	float sum = 0;

	for (int i = 0; i < n_classes; i++) sum += tranformed[begin_idx + i];
	for (int i = 0; i < n_classes; i++) tranformed[begin_idx + i] /= sum;
}

void RandomFernsScan::transformBatch(device_vector<uint8_t>& batch, device_vector<float>& tranformed)
{
	int n_windows = (image_height - win_size)*(image_width - win_size) / stride / stride;
	tranformed = device_vector<float>(batch_size * n_windows * n_classes);
	for (int i = 0; i < ferns.size(); i++) {
		ferns[i].transformBatch(batch, tranformed);
		cudaDeviceSynchronize();
	}

	normalizeTransform<<< 1, dim3(batch_size, n_windows, 1)>>>(tranformed);
}


