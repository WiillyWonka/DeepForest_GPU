#include "fern.cuh"

using std::vector;
using thrust::device_vector;
using thrust::raw_pointer_cast;

Fern::Fern() {}

Fern::Fern(int n_classes, int n_featuress, int depth)
/*
	: depth(depth)
	, n_classes(n_classes)
	, n_features(n_features)
	, h_feature_idx(depth)
	, h_thresholds(depth)
	, h_hist((1 << depth) * n_classes, 1)
	*/
{
	srand(time(0));
	/*
	std::generate(h_feature_idx.begin(), h_feature_idx.end(),
		[=]() { return rand() % n_features; });

	std::generate(h_thresholds.begin(), h_thresholds.end(),
		[=]() { 
			double random = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
			return min_feature + random * (max_feature - min_feature); });
			*/
}

void Fern::moveHost2Device()
{
	d_feature_idx = h_feature_idx;
	d_hist = h_hist;
	d_thresholds = h_thresholds;
	h_feature_idx.clear();
	h_hist.clear();
	h_thresholds.clear();
}

void Fern::moveDevice2Host()
{
	d_feature_idx = h_feature_idx;
	d_hist = h_hist;
	d_thresholds = h_thresholds;
	d_feature_idx.clear();
	d_hist.clear();
	d_thresholds.clear();
}

void Fern::startFitting()
{
	moveHost2Device();
}

void Fern::endFitting()
{
	normalizeHist();
	moveDevice2Host();
}


__global__
void processBatchKernel(
	double* data,
	int* labels,
	int* feature_idx,
	double* thresholds,
	int* hist,
	int n_classes,
	int n_features) 
{
	extern __shared__ char temp[];

	int depth = blockDim.x;
	int idx = feature_idx[threadIdx.x];
	double threshold = thresholds[threadIdx.x];
	double value = data[blockIdx.x * n_features + idx];
	temp[threadIdx.x] = threshold < value;

	if (threadIdx.x != blockDim.x - 1) return;

	int count = 0;
	for (int i = 0; i < depth; i++) count = (count << 1) + temp[i];

	int label = labels[blockIdx.x];
	atomicAdd(&hist[label * (1 << depth) + count], 1);
}

void Fern::processBatch(device_vector<double>& X, device_vector<int>& Y, int batch_size)
{
	//prepare ptrs
	double* data = raw_pointer_cast(X.data());
	int* labels = raw_pointer_cast(Y.data());
	int* feature_idx = raw_pointer_cast(d_feature_idx.data());
	double* thresholds = raw_pointer_cast(d_thresholds.data());
	int* hist = raw_pointer_cast(d_hist.data());

	processBatchKernel <<<1, batch_size, depth * sizeof(char)>>> 
		(data, labels, feature_idx, thresholds, hist, n_classes, n_features);
}

vector<double> Fern::predictProba(vector<double>& X_train)
{
	int idx, threshold, res = 0;
	for (int i = 0; i < h_feature_idx.size(); i++) {
		idx = h_feature_idx[i];
		threshold = h_thresholds[i];
		res = (res << 1) + (threshold < X_train[idx]);
	}

	int step = (1 << depth);
	vector<double> out(n_classes);
	for (int i = 0; i < n_classes; i++) out[i] = h_hist[step * i + res];

	return out;
}

__global__
void normalizeHistKernel(int* hist, int n_classes)
{
	int sum = 0;
	for (int i = 0; i < n_classes; i++) sum += hist[blockDim.x * i + threadIdx.x];

	for (int i = 0; i < n_classes; i++) hist[blockDim.x * i + threadIdx.x] /= sum;
}

void Fern::normalizeHist()
{
	int* hist = raw_pointer_cast(d_hist.data());
	normalizeHistKernel <<<1, (1 << depth) >>>	(hist, n_classes);
}

