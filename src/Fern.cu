#include "fern.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

using std::vector;
using thrust::device_vector;
using thrust::raw_pointer_cast;

Fern::Fern(int n_classes, int n_features, int depth)
	: depth(depth)
	, n_classes(n_classes)
	, n_features(n_features)
	, h_feature_idx(depth)
	, h_thresholds(depth)
	, h_hist((1 << depth)* n_classes)
{
	std::generate(h_feature_idx.begin(), h_feature_idx.end(),
		[=]() { return rand() % n_features; });

	std::generate(h_thresholds.begin(), h_thresholds.end(),
		[=]() { 
			float random = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			return min_feature + random * (max_feature - min_feature); });
}

void Fern::moveHost2Device()
{
	cudaStreamCreate(&stream);
	d_feature_idx = h_feature_idx;
	d_hist = h_hist;
	d_thresholds = h_thresholds;
}

void Fern::releaseDevice()
{
	d_feature_idx.clear();
	d_thresholds.clear();
	d_feature_idx.shrink_to_fit();
	d_thresholds.shrink_to_fit();


	cudaStreamSynchronize(stream);
	d_hist.clear();
	d_hist.shrink_to_fit();

	cudaStreamDestroy(stream);
}

void Fern::startFitting()
{
	cudaStreamCreate(&stream);
	d_feature_idx = h_feature_idx;
	d_thresholds = h_thresholds;
	d_hist = device_vector<float>((1 << depth) * n_classes, 1);
}

void Fern::endFitting()
{
	normalizeHist();
	cudaMemcpyAsync(thrust::raw_pointer_cast(h_hist.data()), thrust::raw_pointer_cast(d_hist.data()),
		d_hist.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
}


__global__
void processBatchKernel(
	float* data,
	uint32_t* labels,
	int* feature_idx,
	float* thresholds,
	float* hist,
	int n_classes,
	int n_features) 
{
	extern __shared__ char temp[];

	int depth = blockDim.x;
	int idx = feature_idx[threadIdx.x];
	float threshold = thresholds[threadIdx.x];
	double value = data[blockIdx.x * n_features + idx];
	temp[threadIdx.x] = threshold < value;

	if (threadIdx.x == blockDim.x - 1) {
		int count = 0;
		for (int i = 0; i < depth; i++) count = (count << 1) + temp[i];

		int label = labels[blockIdx.x];
		atomicAdd(&hist[label * (1 << depth) + count], 1);
	}
}

void Fern::processBatch(device_vector<float>& data, device_vector<uint32_t>& labels)
{
	//prepare ptrs
	float* data_ptr = raw_pointer_cast(data.data());
	uint32_t* labels_ptr = raw_pointer_cast(labels.data());
	int* feature_idx = raw_pointer_cast(d_feature_idx.data());
	float* thresholds = raw_pointer_cast(d_thresholds.data());
	float* hist = raw_pointer_cast(d_hist.data());
	uint32_t batch_size = labels.size();

	processBatchKernel <<<batch_size, depth, depth * sizeof(char), stream >>>
		(data_ptr, labels_ptr, feature_idx, thresholds, hist, n_classes, n_features);
	gpuErrchk(cudaPeekAtLastError());
}

__global__
void transformBatchKernel(
	float* data,
	float* proba,
	int* feature_idx,
	float* thresholds,
	float* hist,
	int n_classes,
	int n_features)
{
	extern __shared__ char temp[];

	int depth = blockDim.x;
	int idx = feature_idx[threadIdx.x];
	float threshold = thresholds[threadIdx.x];
	double value = data[blockIdx.x * n_features + idx];
	temp[threadIdx.x] = threshold < value;

	if (threadIdx.x != blockDim.x - 1) return;

	int count = 0;
	for (int i = 0; i < depth; i++) count = (count << 1) + temp[i];

	for (int i = 0; i < n_classes; i++)
		proba[blockIdx.x * n_classes + i] += hist[(1 << depth) * i + count];
}

void Fern::transformBatch(
	device_vector<float>& data,
	device_vector<float>& proba,
	uint32_t batch_size
	)
{
	float* data_ptr = raw_pointer_cast(data.data());
	float* proba_ptr = raw_pointer_cast(proba.data());
	int* feature_idx = raw_pointer_cast(d_feature_idx.data());
	float* thresholds = raw_pointer_cast(d_thresholds.data());
	float* hist = raw_pointer_cast(d_hist.data());

	transformBatchKernel << <batch_size, depth, depth * sizeof(char), stream >> >
		(data_ptr, proba_ptr, feature_idx, thresholds, hist, n_classes, n_features);
}

vector<float> Fern::predictProbaSingle(vector<double>& X_test)
{
	int idx, threshold, res = 0;
	for (int i = 0; i < h_feature_idx.size(); i++) {
		idx = h_feature_idx[i];
		threshold = h_thresholds[i];
		res = (res << 1) + (threshold < X_test[idx]);
	}

	int step = (1 << depth);
	vector<float> out(n_classes);
	for (int i = 0; i < n_classes; i++) out[i] = h_hist[step * i + res];

	return out;
}

__global__
void normalizeHistKernel(float* hist, int n_classes, int start_idx, int depth)
{
	int hist_size = 1 << depth;
	float sum = 0;
	for (int i = 0; i < n_classes; i++) sum += hist[hist_size * i + start_idx + threadIdx.x];

	for (int i = 0; i < n_classes; i++) hist[hist_size * i + start_idx + threadIdx.x] /= sum;
}

void Fern::normalizeHist()
{
	float* hist = raw_pointer_cast(d_hist.data());

	int max_threads_per_block = 1024;
	int n_threads = std::min(max_threads_per_block, static_cast<int>(d_hist.size() / n_classes));
	for (int i = 0; i < d_hist.size() / n_classes; i += n_threads) {
		n_threads = std::min(max_threads_per_block, static_cast<int>(d_hist.size() - i) / n_classes);
		normalizeHistKernel <<<1, n_threads, 0, stream >>> (hist, n_classes, i, depth);
	}

	gpuErrchk(cudaPeekAtLastError());
}

