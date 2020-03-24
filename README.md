# Linear-Regression-Libtorch

Implementing Linear Regression on a CSV file using PyTorch C++ Frontend API.

**This repository is public but not ready to use. If you want to follow up on our progress, please click** <a href="https://github.com/BuffetCodes/Linear-Regression-using-PyTorch-CPP/milestone/1">here</a>.

## Files

1. `src/main.cpp` - Sample main file to load sample generated data using a numpy-like functions (linspace and random) and perform Linear Regression. 
2. `src/regression.cpp` - Sample main file to load Boston Housing Data and performing Linear Regression.
3. `include/csvloader.h` - Used for `regression.cpp`, has `CSVRow` class to load CSV Files. Credits and Reference: https://stackoverflow.com/a/1120224 (Martin York, Link: https://stackoverflow.com/users/14065/martin-york)
4. `include/utils.h`- Contains helper functions to create and read data.
