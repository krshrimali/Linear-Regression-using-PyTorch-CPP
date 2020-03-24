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

/*
 * This function returns random numbers of length: length, and multiplies each by multiplier
 */
template<typename T>
std::vector<T> random(int length, int multiplier) {
    std::vector<T> vec;

    for (int i = 0; i < length; i++) {
        vec.push_back((rand() % 10) * 2.0);
    }

    return vec;
}

template<typename T>
std::vector<T> add_two_vectors(std::vector<T> a_vector, std::vector<T> b_vector) {
    /*
     * This function adds two vectors and returns the sum vector
     */
    // assert both are of same size
    assert(a_vector.size() == b_vector.size());

    std::vector<T> c_vector;

    for (size_t i = 0; i < a_vector.size(); i++) {
        c_vector.push_back(a_vector[i] + b_vector[i]);
    }

    return c_vector;
}

template<typename T>
std::pair<std::vector<T>, std::vector<T>> create_data() {
    // This creates a data for Linear Regression
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
