#include "FernScan.cuh"

using thrust::device_vector;
using thrust::raw_pointer_cast;

FernScan::FernScan(int win_size, int stride, int depth)
	: depth(depth)
	, win_size(win_size)
	, stride(stride)
	, h_feature_idx(depth)
	, h_thresholds(depth)
{}

__global__
void processBatchKernel(
	uint8_t* data,
	uint32_t* labels,
	uint32_t* feature_idx,
	uint8_t* thresholds,
	float* hist,
	int n_classes,
	int n_features,
	int win_size,
	int stride,
	int height,
	int width)
{
	extern __shared__ char temp[];

	int depth = blockDim.x;
	int idx = feature_idx[threadIdx.x];
	uint8_t value, threshold = thresholds[threadIdx.x];
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			double value = data[blockIdx.x * n_features + y* width + x + (idx / win_size) * width + idx % win_size];
			temp[threadIdx.x] = threshold < value;

			if (threadIdx.x == blockDim.x - 1) {
				int count = 0;
				for (int i = 0; i < depth; i++) count = (count << 1) + temp[i];

				int label = labels[blockIdx.x];
				atomicAdd(&hist[label * (1 << depth) + count], 1);
			}
			__syncthreads();
		}
	}	

	
}

void FernScan::processBatch(device_vector<uint8_t>&data, device_vector<uint32_t>&labels)
{
	//prepare ptrs
	uint8_t* data = raw_pointer_cast(data.data());
	uint32_t* labels = raw_pointer_cast(labels.data());
	uint32_t* feature_idx = raw_pointer_cast(d_feature_idx.data());
	uint8_t* thresholds = raw_pointer_cast(d_thresholds.data());
	float* hist = raw_pointer_cast(d_hist.data());

	processBatchKernel << <batch_size, depth, depth * sizeof(char) >> >
		(data, labels, feature_idx, thresholds, hist, n_classes, n_features,
			win_size, stride, image_height, image_width);
}

void FernScan::moveHost2Device()
{
	d_feature_idx = h_feature_idx;
	d_hist = h_hist;
	d_thresholds = h_thresholds;
	h_hist.clear();
}

void FernScan::releaseDevice()
{
	h_hist = d_hist;
	d_feature_idx.clear();
	d_hist.clear();
	d_thresholds.clear();
}

void FernScan::startFitting()
{
	d_feature_idx = h_feature_idx;
	d_hist = device_vector<float>((1 << depth) * n_classes, 1);
	d_thresholds = h_thresholds;
	h_hist.clear();
}

void FernScan::endFitting()
{
	normalizeHist();
	h_hist = d_hist;
	d_feature_idx.clear();
	d_hist.clear();
	d_thresholds.clear();
}

void FernScan::setClassesNumber(uint32_t n_classes)
{
	this->n_classes = n_classes;
}

void FernScan::setFeaturesNumber(uint32_t n_features)
{
	this->n_features = n_features;
}

__global__
void normalizeHistKernel(float* hist, int n_classes)
{
	float sum = 0;
	for (int i = 0; i < n_classes; i++) sum += hist[blockDim.x * i + threadIdx.x];

	for (int i = 0; i < n_classes; i++) hist[blockDim.x * i + threadIdx.x] /= sum;
}

void FernScan::normalizeHist()
{
	float* hist = raw_pointer_cast(d_hist.data());
	normalizeHistKernel << <1, (1 << depth) >> > (hist, n_classes);
	cudaDeviceSynchronize();
}