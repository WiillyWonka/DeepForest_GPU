#include "DeepForest.h"

DeepForest::DeepForest(const json11::Json& config)
	: scan_cascade(
		config["Scanning Cascade"]["size"].int_value(),
		config["Scanning Cascade"]["N Random Ferns"].int_value(),
		config["Scanning Cascade"]["N Ferns"].int_value(),
		config["Scanning Cascade"]["depth"].int_value(),
		config["Scanning Cascade"]["windows size"].int_value(),
		config["Scanning Cascade"]["stride"].int_value()
	)
{
	if (!config["seed"].is_null())
		srand(config["seed"].int_value());
	else
		srand(time(0));
	
	if (config["Cascade"]["N Random Ferns"].is_null())
		n_random_ferns = 1;
	else
		n_random_ferns = config["Cascade"]["N Random Ferns"].int_value();

	if (config["Cascade"]["N Ferns"].is_null())
		n_ferns = 100;
	else
		n_ferns = config["Cascade"]["N Ferns"].int_value();
	
	if (config["Cascade"]["depth"].is_null())
		depth = 10;
	else
		depth = config["Cascade"]["depth"].int_value();
}

void DeepForest::fit(const vector<vector<uint8_t>>& X, const vector<uint32_t>& y,
	int img_height, int img_width,
	int batch_size)
{
	Timer general_timer;
	std::cout << "Start Deep Forest fitting" << std::endl;
	general_timer.start();

	n_classes = getClassNumber(y);
	n_features = X.begin()->size();

	scan_cascade.setClassesNumber(n_classes);
	scan_cascade.setFeaturesNumber(n_features);

	std::cout << "Fitting of scanning level..." << std::endl;

	Timer timer;
	timer.start();
	scan_cascade.fit(X, y, batch_size);
	timer.stop();

	std::cout << "Fitting time: " << timer.elapsedSeconds() << std::endl;

	std::cout << "Calculating transformed features by scanning level..." << std::endl;

	timer.start();
	scan_cascade.calculateTransform(X, batch_size);
	timer.stop();

	std::cout << "Transformed features calculating time: " << timer.elapsedSeconds() << std::endl;

	double acc = DBL_MAX, prev_acc = 0;

	
	vector<vector<float>> transformed, proba;
	vector<const vector<float>*> X_train, test_transformed;
	vector<uint32_t> train_indices, test_indices, y_train, y_test;

	while (fabs(acc - prev_acc) > tolerance) {
		getKFoldIndices(train_indices, test_indices, y.size());
		transformed = getLastTransformed();

		cascades.push_back(CascadeLevel(n_random_ferns, n_ferns, depth, n_classes, transformed[0].size()));
		CascadeLevel& last_level = cascades.back();

		cascades.back().clearTranformed();

		getSubsetByIndices(transformed, y, train_indices, X_train, y_train);

		std::cout << "Fitting of " << cascades.size() << "th cascade..." << std::endl;

		try {
			timer.start();
			last_level.fit(X_train, y_train, batch_size);
			timer.stop();
		}
		catch (thrust::system::detail::bad_alloc e) {
			throw std::exception("Not enough memory on device");
		}

		std::cout << "Fitting of " << cascades.size() << "th cascade time: " << timer.elapsedSeconds() << std::endl;

		std::cout << "Calculation transformed of " << cascades.size() << "th cascade..." << std::endl;

		try {
			timer.start();
			last_level.caluclateTransform(transformed, batch_size);
			timer.stop();
		}
		catch (thrust::system::detail::bad_alloc e) {
			throw std::exception("Not enough memory on device");
		}

		std::cout << "Calculation transformed of " << cascades.size()
			<< "th cascade time: " << timer.elapsedSeconds() << std::endl;

		std::cout << "Calculating current accuarcy..." << std::endl;

		try {
			timer.start();
			transformed = last_level.getTransfomed();
			timer.stop();
		}
		catch (thrust::system::detail::bad_alloc e) {
			throw std::exception("Not enough memory on device");
		}
		getSubsetByIndices(transformed, y, test_indices, test_transformed, y_test);
		
		proba = probaAveraging(test_transformed);
		
		prev_acc = acc;
		acc = accuracy(y_test, proba);
		
		std::cout << "Calculating current accuarcy time: " << timer.elapsedSeconds() << std::endl;		
		std::cout << "Current accuarcy: " << acc << std::endl;
		
	}

	scan_cascade.clearTransformed();

	for (auto& cascade : cascades)
		cascade.clearTranformed();

	general_timer.stop();
	std::cout << "Deep Forest fitting is over" << std::endl;
	std::cout << "Fitting time: " << general_timer.elapsedSeconds() << std::endl;
}

vector<uint32_t> DeepForest::predict(const vector<vector<uint8_t>>& dataset, int batch_size)
{
	std::cout << "Prediction begins" << std::endl;
	try {
		std::cout << "Calculation scan level" << std::endl;
		scan_cascade.calculateTransform(dataset, batch_size);

		std::cout << "Calculation 1 cascade level" << std::endl;
		cascades.front().caluclateTransform(scan_cascade.getTransformed(0), batch_size);
		CascadeLevel* prev_cascade = &cascades.front();
		
		vector<vector<float>> last_transformed;
		int cascade_idx = 1;
		for (auto it = ++cascades.begin(); it != cascades.end(); it++) {
			std::cout << "Calculation " << cascade_idx + 1 << " cascade level" << std::endl;
			last_transformed = concatenate(prev_cascade->getTransfomed(),
				scan_cascade.getTransformed(cascade_idx % scan_cascade.size()));
			prev_cascade->clearTranformed();
			it->caluclateTransform(last_transformed, batch_size);
			prev_cascade = &(*it);
			cascade_idx++;
		}
		
		vector<vector<float>> proba = probaAveraging(cascades.back().getTransfomed());

		vector<uint32_t> predicted(proba.size());
		for (int i = 0; i < predicted.size(); i++) {
			auto max_proba = std::max_element(proba[i].begin(), proba[i].end());
			predicted[i] = max_proba - proba[i].begin();
		}

		cascades.back().clearTranformed();
		return predicted;
	}
	catch (thrust::system::detail::bad_alloc e) {
		throw std::exception("Not enough memory on device");
	}
	std::cout << "Prediction is over" << std::endl;
}

// This method get last cascade output and calculate probability for all samples
vector<vector<float>> DeepForest::probaAveraging(const vector<vector<float>>& last_output)
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

vector<vector<float>> DeepForest::probaAveraging(const vector<const vector<float>*>& last_output)
{
	size_t output_size = last_output[0]->size();
	vector<vector<float>> out(last_output.size());
	for (int i = 0; i < last_output.size(); i++) {
		out[i] = vector<float>(n_classes, 0);
		for (int j = 0; j < last_output[i]->size(); j++) {
			out[i][j % n_classes] += (*(last_output[i]))[j];
		}

		for (auto& proba : out[i])
			proba /= output_size;
	}

	return out;
}

void DeepForest::getKFoldIndices(
	vector<uint32_t>& train_indices,
	vector<uint32_t>& test_indices,
	size_t dataset_size
	)
{
	assert(dataset_size >= k && "K for k-fold should be greater or equal than size of dataset");

	vector<uint32_t>indices(dataset_size);
	std::iota(indices.begin(), indices.end(), 0);
	std::random_shuffle(indices.begin(), indices.end());

	train_indices = vector<uint32_t>(dataset_size / k * (k - 1));
	int idx;
	for (idx = 0; idx < train_indices.size(); idx++) {
		train_indices[idx] = indices[idx];
	}

	test_indices = vector<uint32_t>(dataset_size - train_indices.size());

	for (int i = 0; i < test_indices.size(); i++) {
		test_indices[i] = indices[i + idx];
	}
}

void DeepForest::getSubsetByIndices(
	const vector<vector<float>>& X_in, const vector<uint32_t>& y_in, const vector<uint32_t>& indices,
	vector<const vector<float>*>& X_out, vector<uint32_t>& y_out)
{
	X_out = vector<const vector<float>*>(indices.size());
	y_out = vector<uint32_t>(indices.size());

	uint32_t index;
	for (uint32_t i = 0; i < indices.size(); i++) {
		index = indices[i];
		X_out[i] = &X_in[index];
		y_out[i] = y_in[index];
	}
}

vector<vector<float>> DeepForest::getLastTransformed()
{
	if (cascades.size() == 0) return scan_cascade.getTransformed(0);

	const vector<vector<float>>& scan_transformed =
		scan_cascade.getTransformed(cascades.size() % scan_cascade.size());
	
	const vector<vector<float>>& last_transformed = cascades.back().getTransfomed();

	return concatenate(last_transformed, scan_transformed);
}

vector<vector<float>> DeepForest::concatenate(const vector<vector<float>>& first, const vector<vector<float>> second)
{
	assert(first.size() == second.size() &&
		"Size of input arrays is not equal");

	vector<vector<float>> out(first.size());

	for (uint32_t j = 0; j < out.size(); j++) {
		out[j] = vector<float>(first[j].size() + second[j].size());
		std::copy(first[j].begin(), first[j].end(), out[j].begin());
		std::copy(second[j].begin(), second[j].end(),
			out[j].begin() + first[j].size());
	}

	return out;
}

uint32_t DeepForest::getClassNumber(const vector<uint32_t>& labels)
{
	return *std::max_element(labels.begin(), labels.end()) + 1;
}

double DeepForest::accuracy(vector<uint32_t>& label, vector<vector<float>>& proba)
{
	assert(label.size() == proba.size() &&
		"proba size should be equal label size");

	uint32_t predicted_class;
	double out = 0;
	for (uint32_t i = 0; i < proba.size(); i++) {
		auto max_proba = std::max_element(proba[i].begin(), proba[i].end());
		predicted_class = max_proba - proba[i].begin();
		out += (label[i] == predicted_class);
	}

	return out / label.size();
}

