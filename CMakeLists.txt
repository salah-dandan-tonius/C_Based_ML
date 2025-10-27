cmake_minimum_required(VERSION 3.14)
project(mymodel)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

add_library(m2c_model STATIC
    Data/Models/DecisionTree_m2c.c      # or the m2c .c file(s)
    # add other C files if needed
)

add_library(mymodule SHARED bindings.cpp)
target_link_libraries(mymodule PRIVATE pybind11::module m2c_model)
set_target_properties(mymodule PROPERTIES PREFIX "")

# ensure position independent code for static lib
set_property(TARGET m2c_model PROPERTY POSITION_INDEPENDENT_CODE ON)
