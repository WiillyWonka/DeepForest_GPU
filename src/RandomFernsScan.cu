#include "RandomFernsScan.cuh"

using std::vector;

void RandomFernsScan::setClassesNumber(uint32_t n_classes)
{
	this->n_classes = n_classes;
	for (auto& fern : ferns) fern.setClassesNumber(n_classes);
}

void RandomFernsScan::setFeaturesNumber(uint32_t n_features)
{
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

void RandomFernsScan::processBatch(thrust::device_vector<uint8_t>& data, thrust::device_vector<uint32_t>& labels)
{
	for (auto& fern : ferns) fern.processBatch(data, labels);
}

void RandomFernsScan::transformBatch(thrust::device_vector<double>& batch, thrust::device_vector<float> tranformed)
{
	int offset;
	for (int i = 0; i < ferns.size(); i++) {
		offset = i * n_classes
		fern.transformBatch(data, labels);

	}
}
