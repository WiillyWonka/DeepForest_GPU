#pragma once

#include <vector>
#include <thrust/device_vector.h>

#include "FernScan.cuh"

class RandomFernsScan
{
public:
	RandomFernsScan(int win_size = 14, int stride = 1, int n_estimators = 3, int depth = 2);

	void setClassesNumber(uint32_t n_classes);
	void setFeaturesNumber(uint32_t n_features);

	void startFitting();
	void endFitting();

	void moveHost2Device();
	void releaseDevice();

	void processBatch(thrust::device_vector<uint8_t>& data, thrust::device_vector<uint32_t>& labels);
	void transformBatch(thrust::device_vector<uint8_t>& batch,
		thrust::device_vector<float>& tranformed, uint32_t batch_size);


private:
	uint32_t image_width = 28, image_height = 28; //TODO: remove hardcode
	uint32_t n_classes, n_features, win_size, stride, n_windows;
	std::vector<FernScan> ferns;

};
