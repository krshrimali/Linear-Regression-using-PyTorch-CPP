// #include <torch/torch.h>
#include <functional> // for placeholders 
#include <vector>
#include <iostream>

std::vector<float> linspace(int start, int end, int length) {
    std::vector<float> vec;
    float diff = (end-start)/float(length);
    for(int i=0; i<length; i++) {
        vec.push_back(start+diff*i);
    }
    return vec;
}

std::vector<float> random(int length, int multiplier) {
    /* 
     * This function returns random numbers of length: length, and multiplies each by multiplier
     */
    std::vector<float> vec;

    for(int i = 0; i < length; i++) {
        vec.push_back( (rand() % 10) * 2.0 );
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

    for(size_t i = 0; i < a_vector.size(); i++) {
        c_vector[i] = a_vector[i] + b_vector[i];
    }

    return c_vector;
}

std::vector<float> create_data() {
    // This creates a data for Linear Regression
    int64_t m = 4; // Slope
    int64_t c = 6; // Intercept
    
    int start = 0;
    int end = 11;
    int length = 91;
    std::vector<float> y = linspace(start, end, length);
   
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
    
    return vec_sum_y;
}

int main() {
    std::vector<float> array = create_data();

    for(size_t i = 0; i < array.size(); i++) {
        std::cout << "Array: " << i << ", : " << array[i] << std::endl;
    }
}
