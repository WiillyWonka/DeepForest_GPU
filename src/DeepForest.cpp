#include "DeepForest.h"

using std::vector;

void DeepForest::fit(const vector<vector<uint8_t>>& X, const vector<uint32_t>& y)
{
	n_classes = getClassNumber(y);
	n_features = X.begin()->size();

	scan_cascade.setClassesNumber(n_classes);
	scan_cascade.setFeaturesNumber(n_features);

	scan_cascade.fit(X, y);

	double acc = 0, prev_acc = DBL_MAX;

	
	vector<vector<float>> transformed;
	vector<vector<uint8_t>&> X_train, X_test;
	vector<uint32_t> y_train, y_test, predicted;

	while (fabs(acc - prev_acc) < tolerance) {
		getKFoldData(X, y, X_train, y_train, X_test, y_test);
		transformed = getLastTransformed(X_train);

		cascade.push_back(CascadeLevel());
		CascadeLevel& last_level = cascade.back();
		last_level.fit(transformed, y_train);

		predicted = predict(X_test);
		prev_acc = acc;
		acc = accuracy(y_test, predicted);
	}
}

void DeepForest::getKFoldData(
	const vector<vector<uint8_t>>& in_X,
	const vector<uint32_t>& in_y,
	vector<vector<uint8_t>&>& train_X,
	vector<uint32_t>& train_y,
	vector<vector<uint8_t>&>& test_X,
	vector<uint32_t>& test_y)
{
	vector<uint32_t> indices(in_X.size());
	std::iota(indices.begin(), indices.end(), 0);

	std::random_shuffle(indices.begin(), indices.end());

	train_X.reserve(in_X.size() * (k - 1) / k);
	train_y.reserve(in_y.size() * (k - 1) / k);

	uint32_t index;
	for (uint32_t i = 0; i < indices.size() * (k - 1) / k; i++) {
		index = indices[i];
		train_X[i] = in_X[index];
		train_y[i] = in_y[index];
	}

	test_X.reserve(in_X.size() / k);
	test_y.reserve(in_y.size() / k);

	uint32_t index;
	for (uint32_t i = indices.size() * (k - 1) / k; i < indices.size(); i++) {
		index = indices[i];
		test_X[i] = in_X[index];
		test_y[i] = in_y[index];
	}
}

vector<vector<float>> DeepForest::getLastTransformed(const vector<vector<uint8_t>&>& data)
{
	vector<vector<float>> scan_transformed, last_transformed, buffer;
	last_transformed = scan_cascade.transform(data, 0);

	if (cascade.size() == 0) return last_transformed;
	
	auto it = cascade.begin();
	for (uint32_t i = 0; i < cascade.size(); i++, it++) {
		scan_transformed = scan_cascade.transform(data, i % scan_cascade.size());
		buffer = it->predict(last_transformed);

		for (uint32_t j = 0; j < buffer.size(); j++) {
			last_transformed[j] = vector<float>(last_transformed[j].size() + scan_transformed[j].size());
			std::copy(buffer[j].begin(), buffer[j].end(), last_transformed[j].begin());
			std::copy(scan_transformed[j].begin(), scan_transformed[j].end(),
				last_transformed[j].begin() + scan_transformed[j].size());
		}

		last_transformed = std::move(buffer);
	}

	return last_transformed;
}

uint32_t DeepForest::getClassNumber(const vector<uint32_t>& labels)
{
	return *std::max_element(labels.begin(), labels.end());
}

double DeepForest::accuracy(vector<uint32_t>& test, vector<uint32_t>& pred)
{
	double out = 0;
	for (uint32_t i = 0; i < test.size(); i++) {
		out += (test[i] == pred[i]);
	}

	return out / test.size();
}
