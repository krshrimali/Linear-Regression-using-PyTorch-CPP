# Linear-Regression-Libtorch

Implementing Linear Regression on a CSV file using PyTorch C++ Frontend API.

**Release:** v0.1 (See <a href="https://github.com/BuffetCodes/Linear-Regression-using-PyTorch-CPP/milestones">Milestone v0.1</a> for more details).

Do check out the video <a href="https://www.youtube.com/watch?v=6raFznPFy2Y">here</a> for Code Review and Demo Run!

## Usage

Follow these steps to implement LR on your CSV file:

1. `git clone https://github.com/BuffetCodes/Linear-Regression-using-PyTorch-CPP.git`.
2. `cd Linear-Regression-using-PyTorch-CPP`.
3. `mkdir build/ && cd build/`.
4. `cmake -DCMAKE_PREFIX_PATH=<absolute_path_to_libtorch>`.
5. `make`.
6. `./bin/lr-example <path to your csv file>` (if you have a CSV file, else do: `./bin/lr-example` to use BostonHousing Data Set in extras folder)

Note that it assumes your last column in the CSV file is the label, and all features are numerical (or can be converted to floating types). The first column (header) is ignored while processing data.

## Files

1. `src/main.cpp` - Sample main file to load sample generated data using a numpy-like functions (linspace and random) and perform Linear Regression.
2. `src/regression.cpp` - Sample main file to load a CSV file and performing Linear Regression.
3. `include/csvloader.h` - Used for `regression.cpp`, has `CSVRow` class to load CSV Files. Credits and Reference: https://stackoverflow.com/a/1120224 (Martin York, Link: https://stackoverflow.com/users/14065/martin-york)
4. `include/utils.h`- Contains helper functions to create, read data and process data.
