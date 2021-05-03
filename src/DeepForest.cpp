#include "DeepForest.h"

DeepForest::DeepForest()
{
	srand(time(0));
}

void DeepForest::fit(const vector<vector<uint8_t>>& X, const vector<uint32_t>& y,
	int img_height, int img_width,
	int batch_size)
{
	n_classes = getClassNumber(y);
	n_features = X.begin()->size();

	scan_cascade.setClassesNumber(n_classes);
	scan_cascade.setFeaturesNumber(n_features);

	scan_cascade.fit(X, y, batch_size);

	double acc = 0, prev_acc = DBL_MAX;

	
	vector<vector<float>> transformed;
	vector<const vector<uint8_t>*> X_train, X_test;
	vector<uint32_t> y_train, y_test, predicted;

	while (fabs(acc - prev_acc) > tolerance) {
		getKFoldData(X, y, X_train, y_train, X_test, y_test);
		transformed = getLastTransformed(X_train, batch_size);

		cascade.push_back(CascadeLevel(n_random_ferns, n_classes, n_features));
		CascadeLevel& last_level = cascade.back();
		last_level.fit(transformed, y_train, batch_size);

		predicted = predict(X_test, batch_size);
		prev_acc = acc;
		acc = accuracy(y_test, predicted);
	}
}

std::vector<uint32_t> DeepForest::predict(const vector<vector<uint8_t>>& X_test, int batch_size)
{
	vector<const vector<uint8_t>*>p_X_test(X_test.size());

	for (int i = 0; i < X_test.size(); i++)
		p_X_test[i] = &X_test[i];

	return predict(p_X_test, batch_size);
}


vector<uint32_t> DeepForest::predict(const vector<const vector<uint8_t>*>& X_test, int batch_size)
{
	vector<vector<float>> proba = probaAveraging(getLastOutput(X_test, batch_size));

	vector<uint32_t> predicted(proba.size());
	int predicted_class;
	for (int i = 0; i < predicted.size(); i++) {
		auto max_proba = std::max_element(proba[i].begin(), proba[i].end());
		predicted[i] = max_proba - proba[i].begin();
	}

	return predicted;
}

vector<vector<float>> DeepForest::probaAveraging(vector<vector<float>> last_output)
{
	vector<vector<float>> out(last_output.size());
	for (int i = 0; i < last_output.size(); i++) {
		out[i] = vector<float>(n_classes, 0);
		for (int j = 0; j < last_output[i].size(); j++) {
			out[i][j % n_classes] += last_output[i][j];
		}

		for (auto& proba : out[i])
			proba /= n_random_ferns;
	}

	return out;
}

void DeepForest::getKFoldData(
	const vector<vector<uint8_t>>& in_X,
	const vector<uint32_t>& in_y,
	vector<const vector<uint8_t>*>& X_test,
	vector<uint32_t>& y_test,
	vector<const vector<uint8_t>*>& X_train,
	vector<uint32_t>& y_train
	)
{
	vector<uint32_t> indices(in_X.size());
	std::iota(indices.begin(), indices.end(), 0);

	std::random_shuffle(indices.begin(), indices.end());

	X_train.reserve(indices.size() * (k - 1) / k);

	uint32_t index;
	for (uint32_t i = 0; i < indices.size() * (k - 1) / k; i++) {
		index = indices[i];
		X_train.push_back(&in_X[index]);
		y_train.push_back(in_y[index]);
	}

	X_test.reserve(in_X.size() / k);
	y_test.reserve(in_y.size() / k);

	for (uint32_t i = indices.size() * (k - 1) / k; i < indices.size(); i++) {
		index = indices[i];
		X_test.push_back(&in_X[index]);
		y_test.push_back(in_y[index]);
	}
}

vector<vector<float>> DeepForest::getLastOutput(const vector<const vector<uint8_t>*>& data, uint32_t batch_size)
{
	vector<vector<float>> scan_transformed, last_transformed, buffer;
	last_transformed = scan_cascade.transform(data, 0, batch_size);

	if (cascade.size() == 0) return last_transformed;

	auto it = cascade.begin();
	for (uint32_t i = 0; i < cascade.size() - 1; i++, it++) {
		scan_transformed = scan_cascade.transform(data, i % scan_cascade.size(), batch_size);
		buffer = it->transform(last_transformed, batch_size);

		for (uint32_t j = 0; j < buffer.size(); j++) {
			last_transformed[j] = vector<float>(last_transformed[j].size() + scan_transformed[j].size());
			std::copy(buffer[j].begin(), buffer[j].end(), last_transformed[j].begin());
			std::copy(scan_transformed[j].begin(), scan_transformed[j].end(),
				last_transformed[j].begin() + scan_transformed[j].size());
		}

		//last_transformed = std::move(buffer);
	}

	last_transformed = cascade.back().transform(last_transformed, batch_size);

	return last_transformed;
}

vector<vector<float>> DeepForest::getLastTransformed(const vector<const vector<uint8_t>*>& data, uint32_t batch_size)
{
	vector<vector<float>> scan_transformed, last_transformed, buffer;
	//std::cout << (cascade.size() - 1) << " " << scan_cascade.size() << " " << (cascade.size() - 1) % scan_cascade.size() << std::endl;
	buffer = getLastOutput(data, batch_size);

	if (cascade.size() == 0) return buffer;

	scan_transformed = scan_cascade.transform(data, cascade.size() % scan_cascade.size(), batch_size);
	for (uint32_t j = 0; j < buffer.size(); j++) {
		last_transformed[j] = vector<float>(last_transformed[j].size() + scan_transformed[j].size());
		std::copy(buffer[j].begin(), buffer[j].end(), last_transformed[j].begin());
		std::copy(scan_transformed[j].begin(), scan_transformed[j].end(),
			last_transformed[j].begin() + scan_transformed[j].size());
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
