#pragma once

#include <vector>
#include <iostream>
#include <algorithm>
#include <cassert>

template<typename T>
std::vector<T> linspace(int start, int end, int length) {
	std::vector<T> vec;
	T diff = (end - start) / T(length);
	for (int i = 0; i < length; i++) {
		vec.push_back(start + diff * i);
	}
	return vec;
}

// This function returns random numbers of length: length, and multiplies each by multiplier
template<typename T>
std::vector<T> random(int length, int multiplier) {
	std::vector<T> vec;

	for (int i = 0; i < length; i++) {
		vec.push_back((rand() % 10) * 2.0);
	}

	return vec;
}

// This function adds two vectors and returns the sum vector
template<typename T>
std::vector<T> add_two_vectors(std::vector<T> a_vector, std::vector<T> b_vector) {
	// assert both are of same size
	assert(a_vector.size() == b_vector.size());

	std::vector<T> c_vector;

	for (size_t i = 0; i < a_vector.size(); i++) {
		c_vector.push_back(a_vector[i] + b_vector[i]);
	}

	return c_vector;
}

// This creates a data for Linear Regression
template<typename T>
std::pair<std::vector<T>, std::vector<T>> create_data() {
	int64_t m = 4; // Slope
	int64_t c = 6; // Intercept

	int start = 0;
	int end = 11;
	int length = 91;
	std::vector<T> y = linspace<T>(start, end, length);
	std::vector<T> x = y;

	// TODO: assert length of y == length
	// Target: y = mx + c + <something random> 

	// Source: https://stackoverflow.com/a/3885136
	// This multiplies the vector with a scalar
	// y = m * x 
	std::transform(y.begin(), y.end(), y.begin(), std::bind1st(std::multiplies<T>(), m));

	// Source: https://stackoverflow.com/a/4461466
	// y = y + c
	std::transform(y.begin(), y.end(), y.begin(), std::bind2nd(std::plus<double>(), c));

	// y = y + <random numbers>
	// There are total 91 numbers
	// y = y + random(91, 2) // calculate 91 random numbers and multiply each by 2
	std::vector<T> random_vector = random<T>(91, 2);
	std::vector<T> vec_sum_y = add_two_vectors<T>(y, random_vector);

	std::pair<std::vector<T>, std::vector<T>>input_output = make_pair(x, vec_sum_y);
	return input_output;
}

// Normalize Feature, Formula: (x - min)/(max - min)
std::vector<float> normalize_feature(std::vector<float> feat) {
	float max_element = *std::max_element(feat.begin(), feat.end());
	float min_element = *std::min_element(feat.begin(), feat.end());
	
	for (int i = 0; i < feat.size(); i++) {
		// max_element - min_element + 1 to avoid divide by zero error
		if(max_element == min_element) {
			feat[i] = (feat[i] - min_element) / (max_element - min_element + 1);
		} else {
			feat[i] = (feat[i] - min_element) / (max_element - min_element);
		}
	}

	return feat;
}

// Linear Regression Model
// Network for Linear Regression. Contains only one Dense Layer
// Usage: auto net = std::make_shared<Net>(1, 1) [Note: Since in_dim = 1, and out_dim = 1]
struct Net : torch::nn::Module {
	Net(int in_dim, int out_dim) {
		fc1 = register_module("fc1", torch::nn::Linear(in_dim, 500));
		fc2 = register_module("fc2", torch::nn::Linear(500, 500));
		fc3 = register_module("fc3", torch::nn::Linear(500, 200));
		fc4 = register_module("fc4", torch::nn::Linear(200, out_dim));
	}

	torch::Tensor forward(torch::Tensor x) {
		x = fc1->forward(x);
		x = fc2->forward(x);
		x = fc3->forward(x);
		x = fc4->forward(x);
		return x;
	}

	torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};
};

// This function processes data, Loads CSV file to vectors and normalizes features to (0, 1)
// Assumes last column to be label and first row to be header (or name of the features)
std::pair<std::vector<float>, std::vector<float>> process_data(std::ifstream&file, CSVRow& row) {
	std::vector<std::vector<float>> features;
	std::vector<float> label;
	int count = 0;
   
	while (file >> row) {
		// Assuming the first row is the name / header of the data set
		if (count == 0) {
			count += 1;
			continue;
		}
		
		for (int i = 0; i < row.size()-1; i++) {
			if(count == 1) {
				// First we initialize each feature vector with a value
				std::vector<float> sample;
				sample.push_back(row[i]);
				features.push_back(sample);
			}
			else {
				features[i].push_back(row[i]);
			}
		}
		
		// Push final column to label vector
		label.push_back(row[row.size()-1]);
		count+= 1;
	}

	// Normalize Features to [0, 1]
	for (int i = 0; i < features.size(); i++) {
		features[i] = normalize_feature(features[i]);
	}

	// Flatten features vectors to 1D
	std::vector<float> inputs = features[0];
	int64_t total = 0;
	for (int i = 1; i < features.size(); i++) {
		total += features[i].size();
	}
	inputs.reserve(total);
	for (int i = 1; i < features.size(); i++) {
		inputs.insert(inputs.end(), features[i].begin(), features[i].end());
	}
	return std::make_pair(inputs, label);
}
