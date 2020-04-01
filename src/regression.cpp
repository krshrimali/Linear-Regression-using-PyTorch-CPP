#include <vector>
#include <iostream>
#include <torch/torch.h>
#include "csvloader.h"
#include "utils.h"
#include <fstream>

int main(int argc, char** argv) {
	// Load CSV data
	std::ifstream file;
	std::string path = argc > 1 ? argv[1] : "../extras/BostonHousing.csv";
	file.open(path, std::ios_base::in);
	// Exit if file not opened successfully
	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		return -1;
	}
	// Process Data, load features and labels for LR
	std::pair<std::vector<float>, std::vector<float>> out = process_data(file);
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

	// Phase 3 : Train and Print Loss
	std::size_t n_epochs = 100;
	for (std::size_t epoch = 1; epoch <= n_epochs; epoch++) {
		auto out = net->forward(input_tensors);
		optimizer.zero_grad();

		auto loss = torch::mse_loss(out, output_tensors);
		float loss_val = loss.item<float>();

		loss.backward();
		optimizer.step();

		std::cout << "Loss: " << loss_val << std::endl;
	}
}
