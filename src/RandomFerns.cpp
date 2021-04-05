#include "RandomFerns.h"

using std::vector;
using thrust::device_vector;

RandomFerns::RandomFerns(int n_classes, int n_features, int n_estimators, int depth)
	: n_classes(n_classes)
	, n_features(n_features)
{
	//TODO: FIX
	ferns.reserve(n_estimators);
	for (auto& fern : ferns) fern = Fern(n_classes, n_features, depth);
}

void RandomFerns::fit(vector<double>& X_train, vector<int>& Y_train, int batch_size)
{
	assert(X_train.size() == Y_train.size());
	device_vector<int> d_Y_train = Y_train;

	for (auto& fern : ferns) fern.startFitting();

	int step = batch_size * n_features;
	device_vector<double> d_data(step);
	device_vector<int> d_labels(batch_size);

	auto it_data = X_train.begin();
	auto it_label = Y_train.begin();
	for (; it_data != X_train.end(); it_data += step) {
		thrust::copy(it_data, it_data + step, d_data.begin());
		thrust::copy(it_label, it_label + step, d_labels.begin());
		for (auto& fern : ferns) fern.processBatch(d_data, d_labels, batch_size);

	}

	for (auto& fern : ferns) fern.endFitting();
}

vector<double> RandomFerns::predictProba(vector<double>& X_test)
{
	/*
	vector<double> fern_proba, proba(n_classes, 0);
	for (auto& fern : ferns) {
		fern_proba = fern.predictProba(X_test);
		for (int i = 0; i < n_classes; i++) proba[i] *= fern_proba[i];
	}

	double sum = 0;
	for (auto& p : proba) sum += p;
	for (auto& p : proba) p /= sum;
	return proba;
	*/
	return vector<double>();
}
