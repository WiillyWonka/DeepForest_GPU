#include "DeepForest.h"

DeepForest::DeepForest(const json11::Json& config)
	: scan_cascade(config["Scanning Cascade"].array_items())
{
	if (!config["seed"].is_null())
		srand(config["seed"].int_value());
	else
		srand(time(0));

	if (!config["cascade size"].is_null())
		cascade_size = config["cascade size"].int_value();
	else
		cascade_size = 5;
	
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

void DeepForest::fit(const vector<p_vector<uint8_t>>& X, const p_vector<uint32_t>& y,
	int img_height, int img_width,
	int batch_size)
{
	Timer general_timer;
	log_debug << "Start Deep Forest fitting" << std::endl;
	general_timer.start();

	n_classes = getClassNumber(y);
	n_features = X.begin()->size();

	scan_cascade.setClassesNumber(n_classes);
	scan_cascade.setFeaturesNumber(n_features);

	log_debug << "Fitting of scanning level..." << std::endl;

	Timer timer;
	timer.start();
	scan_cascade.fit(X, y, batch_size);
	timer.stop();

	log_debug << "Fitting time: " << timer.elapsedSeconds() << std::endl;

	log_debug << "Calculating transformed features by scanning level..." << std::endl;

	timer.start();
	scan_cascade.calculateTransform(X, batch_size);
	timer.stop();

	log_debug << "Transformed features calculating time: " << timer.elapsedSeconds() << std::endl;

	double acc = DBL_MAX, prev_acc = 0;

	
	vector<p_vector<float>> transformed;
	vector<vector<float>> proba;
	vector<const p_vector<float>*> X_train, test_transformed;
	vector<uint32_t> train_indices, test_indices;
	p_vector<uint32_t> y_train, y_test;

	while (fabs(acc - prev_acc) > tolerance && cascades.size() < cascade_size) {
		getKFoldIndices(train_indices, test_indices, y.size());
		transformed = getLastTransformed();

		cascades.push_back(CascadeLevel(n_random_ferns, n_ferns, depth, n_classes, transformed[0].size()));
		CascadeLevel& last_level = cascades.back();

		cascades.back().clearTranformed();

		getSubsetByIndices(transformed, y, train_indices, X_train, y_train);

		log_debug << "Fitting of " << cascades.size() << "th cascade..." << std::endl;

		try {
			timer.start();
			last_level.fit(X_train, y_train, batch_size);
			timer.stop();
		}
		catch (thrust::system::detail::bad_alloc e) {
			throw std::exception("Not enough memory on device");
		}

		log_debug << "Fitting of " << cascades.size() << "th cascade time: " << timer.elapsedSeconds() << std::endl;

		log_debug << "Calculation transformed of " << cascades.size() << "th cascade..." << std::endl;

		try {
			timer.start();
			last_level.calculateTransform(transformed, batch_size);
			timer.stop();
		}
		catch (thrust::system::detail::bad_alloc e) {
			throw std::exception("Not enough memory on device");
		}

		log_debug << "Calculation transformed of " << cascades.size()
			<< "th cascade time: " << timer.elapsedSeconds() << std::endl;

		log_debug << "Calculating current accuarcy..." << std::endl;

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
		
		log_debug << "Calculating current accuarcy time: " << timer.elapsedSeconds() << std::endl;
		log_debug << "Current accuarcy: " << acc << std::endl;
		
	}
	cudaProfilerStop();
 	scan_cascade.clearTransformed();

	for (auto& cascade : cascades)
		cascade.clearTranformed();

	general_timer.stop();
	log_debug << "Deep Forest fitting is over" << std::endl;
	log_debug << "Fitting time: " << general_timer.elapsedSeconds() << std::endl;
}

vector<uint32_t> DeepForest::predict(const vector<p_vector<uint8_t>>& dataset, int batch_size)
{
	try {
		scan_cascade.calculateTransform(dataset, batch_size);

		cascades.front().calculateTransform(scan_cascade.getTransformed(0), batch_size);
		CascadeLevel* prev_cascade = &cascades.front();
		
		vector<p_vector<float>> last_transformed;
		int cascade_idx = 1;
		for (auto it = ++cascades.begin(); it != cascades.end(); it++) {
			last_transformed = concatenate(prev_cascade->getTransfomed(),
				scan_cascade.getTransformed(cascade_idx % scan_cascade.size()));
			prev_cascade->clearTranformed();
			it->calculateTransform(last_transformed, batch_size);
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
}

// This method get last cascade output and calculate probability for all samples
vector<vector<float>> DeepForest::probaAveraging(const vector<p_vector<float>>& last_output)
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

vector<vector<float>> DeepForest::probaAveraging(const vector<const p_vector<float>*>& last_output)
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
	const vector<p_vector<float>>& X_in, const p_vector<uint32_t>& y_in, const vector<uint32_t>& indices,
	vector<const p_vector<float>*>& X_out, p_vector<uint32_t>& y_out)
{
	X_out = vector<const p_vector<float>*>(indices.size());
	y_out = vector<uint32_t>(indices.size());

	uint32_t index;
	for (uint32_t i = 0; i < indices.size(); i++) {
		index = indices[i];
		X_out[i] = &X_in[index];
		y_out[i] = y_in[index];
	}
}

vector<p_vector<float>> DeepForest::getLastTransformed()
{
	if (cascades.size() == 0) return scan_cascade.getTransformed(0);

	const vector<p_vector<float>>& scan_transformed =
		scan_cascade.getTransformed(cascades.size() % scan_cascade.size());
	
	const vector<p_vector<float>>& last_transformed = cascades.back().getTransfomed();

	return concatenate(last_transformed, scan_transformed);
}

vector<p_vector<float>> DeepForest::concatenate(
	const vector<p_vector<float>>& first, const vector<p_vector<float>> second)
{
	assert(first.size() == second.size() &&
		"Size of input arrays is not equal");

	vector<p_vector<float>> out(first.size());

	for (uint32_t j = 0; j < out.size(); j++) {
		out[j] = vector<float>(first[j].size() + second[j].size());
		std::copy(first[j].begin(), first[j].end(), out[j].begin());
		std::copy(second[j].begin(), second[j].end(),
			out[j].begin() + first[j].size());
	}

	return out;
}

uint32_t DeepForest::getClassNumber(const p_vector<uint32_t>& labels)
{
	return *std::max_element(labels.begin(), labels.end()) + 1;
}

double DeepForest::accuracy(p_vector<uint32_t>& label, vector<vector<float>>& proba)
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

