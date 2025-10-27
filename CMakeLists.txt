cmake_minimum_required(VERSION 3.14)
project(mymodule)

# Require Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Create a static library for all m2cgen C models
add_library(m2c_models STATIC
    Data/Models/DecisionTree_m2c.c
    Data/Models/RandomForest_m2c.c
    Data/Models/LogisticRegression_m2c.c
)

# Ensure position-independent code for static lib
set_property(TARGET m2c_models PROPERTY POSITION_INDEPENDENT_CODE ON)

# Build the Python module
add_library(mymodule SHARED bindings.cpp)
target_link_libraries(mymodule PRIVATE pybind11::module m2c_models)

# Remove the default 'lib' prefix on the shared library
set_target_properties(mymodule PROPERTIES PREFIX "")

# Optional: set C++ standard
set_target_properties(mymodule PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
