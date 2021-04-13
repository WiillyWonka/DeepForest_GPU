#pragma once

#include <vector>
#include <thrust/device_vector.h>

#include "FernScan.cuh"

class RandomFernsScan
{
public:
	RandomFernsScan(int win_size, int stride, int batch_size, int n_estimators = 100, int depth = 10);

	void setClassesNumber(uint32_t n_classes);
	void setFeaturesNumber(uint32_t n_features);

	void startFitting();
	void endFitting();

	void moveHost2Device();
	void releaseDevice();

	void processBatch(thrust::device_vector<uint8_t>& data, thrust::device_vector<uint32_t>& labels);
	void transformBatch(thrust::device_vector<uint8_t>& batch, thrust::device_vector<float>& tranformed);


private:
	uint32_t n_classes, n_features, win_size, stride, batch_size;
	uint32_t image_width = 28, image_height = 28; //TODO: remove hardcode
	std::vector<FernScan> ferns;

};
