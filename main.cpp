// #include <torch/torch.h>
#include <functional> // for placeholders 
#include <vector>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <torch/torch.h>

std::vector<float> linspace(int start, int end, int length) {
    std::vector<float> vec;
    float diff = (end - start) / float(length);
    for (int i = 0; i < length; i++) {
        vec.push_back(start + diff * i);
    }
    return vec;
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
        fc1 = register_module("fc1", torch::nn::Linear(in_dim, out_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = fc1->forward(x);
        return x;
    }

    torch::nn::Linear fc1{ nullptr };
};

int main() {
    // std::vector<float> inputs, std::vector<float> outputs = create_data();
    std::pair<std::vector<float>, std::vector<float>> pair_input_output = create_data();
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