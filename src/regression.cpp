#include <functional> // for placeholders 
#include <vector>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <torch/torch.h>
#include "csvloader.h"

std::vector<float> linspace(int start, int end, int length) {
	std::vector<float> vec;
	float diff = (end - start) / float(length);
	for (int i = 0; i < length; i++) {
		vec.push_back(start + diff * i);
	}
	return vec;
}

// (x - min)/(max - min)
std::vector<float> normalize_feature(std::vector<float> feat) {
	std::cout << feat.size() << std::endl;
    float max_element = *std::max_element(feat.begin(), feat.end());
	float min_element = *std::min_element(feat.begin(), feat.end());
    
	for (int i = 0; i < feat.size(); i++) {
		feat[i] = (feat[i] - min_element) / (max_element - min_element);
	}

	return feat;
}

std::vector<float> random(int length, int multiplier) {
	/*
	 * This function returns random numbers of length: length, and multiplies each by multiplier
	 */
	std::vector<float> vec;

	for (int i = 0; i < length; i++) {
		vec.push_back((rand() % 10) * 2.0);
	}

	return vec;
}

std::vector<float> add_two_vectors(std::vector<float> a_vector, std::vector<float> b_vector) {
	/*
	 * This function adds two vectors and returns the sum vector
	 */
	 // assert both are of same size
	assert(a_vector.size() == b_vector.size());

	std::vector<float> c_vector;

	for (size_t i = 0; i < a_vector.size(); i++) {
		c_vector.push_back(a_vector[i] + b_vector[i]);
	}

	return c_vector;
}

std::pair<std::vector<float>, std::vector<float>> sample_data() {
	int start = 0;
	int end = 20;
	int length = 91;
	std::vector<float> random_vector = random(91, 2);
	std::vector<float> input_vector = add_two_vectors(linspace(start, end, length), random(91, 4));

	std::pair<std::vector<float>, std::vector<float>>input_output = std::make_pair(input_vector, random_vector);
}

std::pair<std::vector<float>, std::vector<float>> create_data() {
	// This creates a data for Linear Regression
	int64_t m = 4; // Slope
	int64_t c = 6; // Intercept

	int start = 0;
	int end = 11;
	int length = 91;
	std::vector<float> y = linspace(start, end, length);
	std::vector<float> x = y;

	// TODO: assert length of y == length
	// Target: y = mx + c + <something random> 

	// Source: https://stackoverflow.com/a/3885136
	// This multiplies the vector with a scalar
	// y = m * x 
	std::transform(y.begin(), y.end(), y.begin(), std::bind1st(std::multiplies<float>(), m));

	// Source: https://stackoverflow.com/a/4461466
	// y = y + c
	std::transform(y.begin(), y.end(), y.begin(), std::bind2nd(std::plus<double>(), c));

	// y = y + <random numbers>
	// There are total 91 numbers
	// y = y + random(91, 2) // calculate 91 random numbers and multiply each by 2
	std::vector<float> random_vector = random(91, 2);
	std::vector<float> vec_sum_y = add_two_vectors(y, random_vector);

	std::pair<std::vector<float>, std::vector<float>>input_output = make_pair(x, vec_sum_y);
	return input_output;
}

// Linear Regression Model
struct Net : torch::nn::Module {
	/*
	Network for Linear Regression. Contains only one Dense Layer
	Usage: auto net = std::make_shared<Net>(1, 1) [Note: Since in_dim = 1, and out_dim = 1]
	*/
	Net(int in_dim, int out_dim) {
		fc1 = register_module("fc1", torch::nn::Linear(in_dim, 100));
	    fc2 = register_module("fc2", torch::nn::Linear(100, out_dim));
    }

	torch::Tensor forward(torch::Tensor x) {
		x = fc1->forward(x);
        x = fc2->forward(x);
		return x;
	}

	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };
};

int main() {
	// Uncomment three lines below if you want to load a sample random data
	// std::vector<float> inputs, std::vector<float> outputs = create_data();
	// std::pair<std::vector<float>, std::vector<float>> pair_input_output = create_data();
	// std::vector<float> inputs = pair_input_output.first;
	// std::vector<float> outputs = pair_input_output.second;

	// Load CSV data
	// TODO: Add an assert here
	std::ifstream file("extras/BostonHousing.csv");
	CSVRow	row;
	
	std::vector<float> crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, B, lstat, medv;

	int count = 0;
	while (file >> row) {
		if (count++ == 0) continue;
        std::cout << row[0] << ", " << row[1] << ", " << row[2] << ", " << ", " << row[4] << std::endl;
		/*
		for (int i = 0; i <= 13; i++) {
			std::cout << row[i] << ", ";
			std::cout << stof(row[i]) << ", ";
		}
		*/
		// std::cout << ", " << row[0] << std::endl;
		crim.push_back(row[0]);
		zn.push_back(row[1]);
		indus.push_back(row[2]);
		// chas.push_back(row[3]);
		nox.push_back(row[4]);
		rm.push_back(row[5]);
		age.push_back(row[6]);
		dis.push_back(row[7]);
		rad.push_back(row[8]);
		tax.push_back(row[9]);
		ptratio.push_back(row[10]);
		B.push_back(row[11]);
		lstat.push_back(row[12]);
		medv.push_back(row[13]);

	}
    
	crim = normalize_feature(crim);
	zn = normalize_feature(zn);
	indus = normalize_feature(indus);
	nox = normalize_feature(nox);
	rm = normalize_feature(rm);
	age = normalize_feature(age);
	dis = normalize_feature(dis);
	rad = normalize_feature(rad);
	tax = normalize_feature(tax);
	ptratio = normalize_feature(ptratio);
	B = normalize_feature(B);
	lstat = normalize_feature(lstat);
    

    // medv = normalize_feature(medv);

	// std::cout << crim.size() << std::endl;
	std::vector<float> inputs = crim;
	inputs.reserve(crim.size() + zn.size() + indus.size() + nox.size() + rm.size() + age.size() + \
		dis.size() + rad.size() + tax.size() + ptratio.size() + B.size() + lstat.size());
	// inputs.insert(inputs.end(), crim.begin(), crim.end());
	inputs.insert(inputs.end(), zn.begin(), zn.end());
	inputs.insert(inputs.end(), indus.begin(), indus.end());
	// inputs.insert(inputs.end(), chas.begin(), chas.end());
	inputs.insert(inputs.end(), nox.begin(), nox.end());
	inputs.insert(inputs.end(), rm.begin(), rm.end());
	inputs.insert(inputs.end(), age.begin(), age.end());
	inputs.insert(inputs.end(), dis.begin(), dis.end());
	inputs.insert(inputs.end(), rad.begin(), rad.end());
	inputs.insert(inputs.end(), tax.begin(), tax.end());
	inputs.insert(inputs.end(), ptratio.begin(), ptratio.end());
	inputs.insert(inputs.end(), B.begin(), B.end());
	inputs.insert(inputs.end(), lstat.begin(), lstat.end());

	std::vector<float> outputs = medv;
	
	// Phase 1 : Data created
	for (size_t i = 0; i < outputs.size(); i++) {
		std::cout << "Outputs: " << i << ", : " << outputs[i] << std::endl;
	}

	// Convert array to a tensor
	// Each should be float32?
	// Reference: https://discuss.pytorch.org/t/passing-stl-container-to-torch-tensors/36614/2

	auto output_tensors = torch::from_blob(outputs.data(), { 506, 1 });
	auto input_tensors = torch::from_blob(inputs.data(), { 506, 12 });
	
	std::cout << input_tensors.sizes() << std::endl;
	std::cout << output_tensors.sizes() << std::endl;

	// Phase 2 : Create Network
	auto net = std::make_shared<Net>(12, 1);
	torch::optim::SGD optimizer(net->parameters(), 0.001);

	for (size_t epoch = 1; epoch <= 1000; epoch++) {
		auto out = net->forward(input_tensors);
		optimizer.zero_grad();

		// auto loss = torch::smooth_l1_loss(out, output_tensors);
		auto loss = torch::mse_loss(out, output_tensors);
		float loss_val = loss.item<float>();

		loss.backward();
		optimizer.step();

		std::cout << "Loss: " << loss_val << std::endl;
	}
}
