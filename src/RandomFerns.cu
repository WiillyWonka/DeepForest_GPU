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
	srand(time(0));
	ferns.reserve(n_estimators);
	for (int i = 0; i < n_estimators; i++) ferns.push_back(Fern(n_classes, n_features, depth));
}

void RandomFerns::fit(vector<double>& X_train, vector<int>& Y_train, int batch_size)
{
	std::cout << std::endl;
	for (auto& fern : ferns) fern.startFitting();

	int data_step = batch_size * n_features;
	device_vector<double> d_data(data_step);
	device_vector<int> d_labels(batch_size);

	auto it_data = X_train.begin();
	auto it_label = Y_train.begin();
	while (it_data != X_train.end()) {

		try
		{
			thrust::copy(it_data, it_data + data_step, d_data.begin());
			thrust::copy(it_label, it_label + batch_size, d_labels.begin());
		}
		catch (thrust::system_error e)
		{
			std::cerr << "Error inside copy: " << e.what() << std::endl;
			return;
		}

		for (auto& fern : ferns) fern.processBatch(d_data, d_labels, batch_size);
		it_data += data_step;
		it_label += batch_size;
		gpuErrchk(cudaDeviceSynchronize());
	}

	for (auto& fern : ferns) fern.endFitting();
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

device_vector<float> RandomFerns::predictProba(std::vector<double>& X_test, int batch_size)
{
	for (auto& fern : ferns) fern.moveHost2Device();

	int data_step = batch_size * n_features;
	int proba_step = batch_size * n_classes;
	int n_samples = X_test.size() / n_features;
	device_vector<double> d_data(data_step);
	device_vector<float> proba(n_samples * n_classes);
	thrust::device_ptr<float> proba_ptr = proba.data();

	auto it_data = X_test.begin();
	while (it_data != X_test.end()) {

		try
		{
			thrust::copy(it_data, it_data + data_step, d_data.begin());
		}
		catch (thrust::system_error e)
		{
			std::cerr << "Error inside copy: " << e.what() << std::endl;
			return device_vector<float>();
		}

		for (auto& fern : ferns) fern.predictProbaBatch(d_data, proba_ptr, batch_size);
		it_data += data_step;
		proba_ptr += proba_step;
		gpuErrchk(cudaDeviceSynchronize());
	}

	for (auto& fern : ferns) fern.moveDevice2Host();

	normalizeProba << < 1, n_samples >>> (raw_pointer_cast(proba.data()), n_classes);

	return proba;
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
