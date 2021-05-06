#include "FernScan.cuh"

using thrust::device_vector;
using thrust::raw_pointer_cast;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

FernScan::FernScan(int win_size, int stride, int depth)
	: depth(depth)
	, win_size(win_size)
	, stride(stride)
	, h_feature_idx(depth)
	, h_thresholds(depth)
{

	std::generate(h_feature_idx.begin(), h_feature_idx.end(),
		[=]() { return rand() % (win_size * win_size); });

	std::generate(h_thresholds.begin(), h_thresholds.end(),
		[=]() {	return rand() % max_feature; });
}

__global__
void processBatchKernelScan(
	unsigned char* data,
	unsigned int* labels,
	int* feature_idx,
	int* thresholds,
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
	int value, threshold = thresholds[threadIdx.x];
	for (int x = 0; x < width - win_size + 1; x += stride) {
		for (int y = 0; y < height - win_size + 1; y += stride) {
				
			value = data[blockIdx.x * n_features + y* width + x + (idx / win_size) * width + idx % win_size];
			temp[threadIdx.x] = threshold < value;

			if (threadIdx.x == blockDim.x - 1) {
				int count = 0;
				for (int i = 0; i < depth; i++)
					count = (count << 1) + temp[i];

				unsigned int label = labels[blockIdx.x];

				atomicAdd(&hist[label * (1 << depth) + count], 1);
			}
			__syncthreads();
		}
	}	

	
}

void FernScan::processBatch(device_vector<uint8_t>& data, device_vector<uint32_t>& labels)
{
	//prepare ptrs
	unsigned char* data_ptr = raw_pointer_cast(data.data());
	unsigned int* labels_ptr = raw_pointer_cast(labels.data());
	int* feature_idx = raw_pointer_cast(d_feature_idx.data());
	int* thresholds = raw_pointer_cast(d_thresholds.data());
	float* hist = raw_pointer_cast(d_hist.data());
	int batch_size = labels.size();
	
	processBatchKernelScan << <batch_size, depth, depth * sizeof(char) >> >
		(data_ptr, labels_ptr, feature_idx, thresholds, hist, n_classes, n_features,
			win_size, stride, image_height, image_width);

	gpuErrchk(cudaPeekAtLastError());
}

__global__
void transformBatchKernelScan(
	uint8_t* data,
	float* transformed,
	int* feature_idx,
	int* thresholds,
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
	int value, threshold = thresholds[threadIdx.x];
	for (int x = 0; x < width - win_size + 1; x += stride) {
		for (int y = 0; y < height - win_size + 1; y += stride) {
			value = data[blockIdx.x * n_features + y * width + x + (idx / win_size) * width + idx % win_size];
			temp[threadIdx.x] = threshold < value;

			if (threadIdx.x == blockDim.x - 1) {
				int count = 0;
				for (int i = 0; i < depth; i++)
					count = (count << 1) + temp[i];

				int n_windows = (height - win_size + 1) * (width - win_size + 1) / stride / stride;
				int sample_size = n_windows * n_classes;
				int start_idx = sample_size * blockIdx.x + (x + y) * n_classes;
				for (int i = 0; i < n_classes; i++)
					transformed[start_idx + i] += hist[(1 << depth) * i + count];
			}
			__syncthreads();
		}
	}


}

void FernScan::transformBatch(device_vector<uint8_t>& data, device_vector<float>& transformed, uint32_t batch_size)
{
	uint8_t* data_ptr = raw_pointer_cast(data.data());
	float* transformed_ptr = raw_pointer_cast(transformed.data());
	int* feature_idx = raw_pointer_cast(d_feature_idx.data());
	int* thresholds = raw_pointer_cast(d_thresholds.data());
	float* hist = raw_pointer_cast(d_hist.data());

	transformBatchKernelScan << <batch_size, depth, depth * sizeof(char) >> >
		(data_ptr, transformed_ptr, feature_idx, thresholds, hist, n_classes, n_features,
			win_size, stride, image_height, image_width);

	gpuErrchk(cudaPeekAtLastError());
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
void normalizeHistScanKernel(float* hist, int n_classes, int start_idx, int depth)
{
	int hist_size = 1 << depth;
	float sum = 0;
	for (int i = 0; i < n_classes; i++) sum += hist[hist_size * i + start_idx + threadIdx.x];

	for (int i = 0; i < n_classes; i++) hist[hist_size * i + start_idx + threadIdx.x] /= sum;
}

void FernScan::normalizeHist()
{
	float* hist = raw_pointer_cast(d_hist.data());

	int max_threads_per_block = 512;
	int n_threads = std::min(max_threads_per_block, static_cast<int>(d_hist.size() / n_classes));
	for (int i = 0; i < d_hist.size() / n_classes; i += n_threads) {
		n_threads = std::min(max_threads_per_block, static_cast<int>((d_hist.size() - i) / n_classes));
		normalizeHistScanKernel << <1, n_threads >> > (hist, n_classes, i, depth);
	}

	gpuErrchk(cudaPeekAtLastError());
}