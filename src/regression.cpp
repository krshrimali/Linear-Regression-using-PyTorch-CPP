#include <vector>
#include <iostream>
#include <torch/torch.h>
#include "csvloader.h"
#include <fstream>

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

    max_element = *std::max_element(feat.begin(), feat.end());
    min_element = *std::min_element(feat.begin(), feat.end());

	return feat;
}

// Linear Regression Model
// Network for Linear Regression. Contains only one Dense Layer
// Usage: auto net = std::make_shared<Net>(1, 1) [Note: Since in_dim = 1, and out_dim = 1]
struct Net : torch::nn::Module {
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

int main(int argc, char** argv) {
	// Load CSV data
	std::ifstream file;
    CSVRow row;
    if (argc > 1) {
        file.open("../extras/BostonHousing.csv", std::ios_base::in);
    } else {
        file.open("../extras/BostonHousing.csv", std::ios_base::in);
    }
    
    // Process Data, load features and labels for LR
	std::pair<std::vector<float>, std::vector<float>> out = process_data(file, row);
	std::vector<float> inputs = out.first;
	std::vector<float> outputs = out.second;
    
	// Phase 1 : Data created
	
    // Convert vectors to a tensor
	// Reference: https://discuss.pytorch.org/t/passing-stl-container-to-torch-tensors/36614/2
    
    // These fields should not be hardcoded (506, 1, 13)
	auto output_tensors = torch::from_blob(outputs.data(), {int(outputs.size()), 1});
	auto input_tensors = torch::from_blob(inputs.data(), {int(outputs.size()), int(inputs.size()/outputs.size())});

	// Phase 2 : Create Network
	auto net = std::make_shared<Net>(int(input_tensors.sizes()[1]), 1);
	torch::optim::SGD optimizer(net->parameters(), 0.001);

	int n_epochs = 100;
	for (size_t epoch = 1; epoch <= n_epochs; epoch++) {
		auto out = net->forward(input_tensors);
        optimizer.zero_grad();

		auto loss = torch::mse_loss(out, output_tensors);
		float loss_val = loss.item<float>();

		loss.backward();
		optimizer.step();

		std::cout << "Loss: " << loss_val << std::endl;
	}
}
