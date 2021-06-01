#include "RandomFernsScan.cuh"

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

RandomFernsScan::RandomFernsScan(int win_size, int stride, int n_estimators, int depth)
	: win_size(win_size)
	, stride(stride)
	, n_windows(((image_height - win_size) / stride + 1) *
		((image_width - win_size)  / stride + 1))
{
	srand(time(0));
	ferns.reserve(n_estimators);
	for (int i = 0; i < n_estimators; i++) ferns.push_back(FernScan(win_size, stride, depth));
}

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
	for (auto& fern : ferns)
		fern.processBatch(data, labels);
}

__global__
void normalizeTransform(float* tranformed, int n_windows, int n_classes)
{
	int sample_idx = blockIdx.x;
	int win_idx = threadIdx.x;
	int begin_idx = sample_idx * n_windows * n_classes + win_idx * n_classes;

	float sum = 0;

	for (int i = 0; i < n_classes; i++) sum += tranformed[begin_idx + i];
	for (int i = 0; i < n_classes; i++) tranformed[begin_idx + i] /= sum;
}

void RandomFernsScan::transformBatch(device_vector<uint8_t>& batch,
	device_vector<float>& tranformed, uint32_t batch_size)
{
	tranformed = device_vector<float>(batch_size * n_windows * n_classes, 0);
	for (int i = 0; i < ferns.size(); i++) {
		ferns[i].transformBatch(batch, tranformed, batch_size);
		cudaDeviceSynchronize();
	}

	normalizeTransform<<< batch_size, n_windows >>>(
		thrust::raw_pointer_cast(tranformed.data()),
		n_windows, n_classes);
	gpuErrchk(cudaPeekAtLastError());
}


