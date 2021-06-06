#include "RandomFerns.cuh"

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

RandomFerns::RandomFerns(int n_classes, int n_features, int n_estimators, int depth)
	: n_classes(n_classes)
	, n_features(n_features)
{
	ferns.reserve(n_estimators);
	for (int i = 0; i < n_estimators; i++) ferns.push_back(Fern(n_classes, n_features, depth));
}

void RandomFerns::processBatch(device_vector<float>& data, device_vector<uint32_t>& labels)
{
	for (auto& fern : ferns) fern.processBatch(data, labels);
	//gpuErrchk(cudaDeviceSynchronize());
	//ferns[0].printValue();
}

__global__
void normalizeProba(float* proba, int n_classes) {
	proba += n_classes * threadIdx.x;

	float sum = 0;
	for (int i = 0; i < n_classes; i++)
		sum += proba[i];

	for (int i = 0; i < n_classes; i++)
		proba[i] /= sum;
}


vector<vector<float>> RandomFerns::transformBatch(thrust::device_vector<float>& data, uint32_t batch_size)
{

	device_vector<float> proba(batch_size * n_classes, 0);

	for (auto& fern : ferns) fern.transformBatch(data, proba, batch_size);

	cudaDeviceSynchronize();

	normalizeProba << < 1, batch_size >> > (raw_pointer_cast(proba.data()), n_classes);
	gpuErrchk(cudaPeekAtLastError());
	cudaDeviceSynchronize();

	return unpackProba(proba);
}


vector<float> RandomFerns::predictProbaSingle(vector<double>& X_test)
{
	vector<float> fern_proba, proba(n_classes, 0);
	for (auto& fern : ferns) {
		fern_proba = fern.predictProbaSingle(X_test);
		for (int i = 0; i < n_classes; i++) proba[i] += fern_proba[i];
	}

	double sum = 0;
	for (auto& p : proba) sum += p;
	for (auto& p : proba) p /= sum;
	return proba;
}

void RandomFerns::startFitting()
{
	for (auto& fern : ferns) fern.startFitting();
}

void RandomFerns::endFitting()
{
	for (auto& fern : ferns) fern.endFitting();
}

void RandomFerns::moveHost2Device()
{
	for (auto& fern : ferns) fern.moveHost2Device();
}

void RandomFerns::releaseDevice()
{
	for (auto& fern : ferns) fern.releaseDevice();
}

vector<vector<float>> RandomFerns::unpackProba(device_vector<float>& proba)
{
	uint32_t batch_size = proba.size() / n_classes;

	vector<vector<float>> out(batch_size);

	for (int i = 0; i < batch_size; i++) {
		out[i] = vector<float>(n_classes);
		auto it = proba.begin() + i * n_classes;
		thrust::copy(it, it + n_classes, out[i].begin());
	}

	return out;
}
