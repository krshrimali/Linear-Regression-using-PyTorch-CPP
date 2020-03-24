#include <functional> // for placeholders
#include "utils.h"
#include <torch/torch.h>

/* TODO: this needs to go in header file */
// Linear Regression Model
struct Net : torch::nn::Module {
    /*
    Network for Linear Regression. Contains only one Dense Layer
    Usage: auto net = std::make_shared<Net>(1, 1) [Note: Since in_dim = 1, and out_dim = 1]
    */
    Net(int in_dim, int out_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(in_dim, out_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = fc1->forward(x);
        return x;
    }

    torch::nn::Linear fc1{ nullptr };
};

/* TODO: This stays */
int main() {
    // std::vector<float> inputs, std::vector<float> outputs = create_data();
    std::pair<std::vector<float>, std::vector<float>> pair_input_output = create_data<float>();
    std::vector<float> inputs = pair_input_output.first;
    std::vector<float> outputs = pair_input_output.second;

    // Phase 1 : Data created
    for (size_t i = 0; i < outputs.size(); i++) {
        std::cout << "Outputs: " << i << ", : " << outputs[i] << std::endl;
    }
    
    // x.reshape(-1, 1)
    // y.reshape(-1, 1)
    
    // Convert array to a tensor
    // Each should be float32?
    // Reference: https://discuss.pytorch.org/t/passing-stl-container-to-torch-tensors/36614/2

    auto output_tensors = torch::from_blob(outputs.data(), { 91, 1});
    auto input_tensors = torch::from_blob(inputs.data(), { 91, 1});

    // Phase 2 : Create Network
    auto net = std::make_shared<Net>(1, 1);
    torch::optim::SGD optimizer(net->parameters(), 0.012);
    
    for (size_t epoch = 1; epoch <= 100; epoch++) {
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
