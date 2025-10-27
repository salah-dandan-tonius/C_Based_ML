#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

// Include all generated C models
extern "C" void score_DecisionTree(double *input, double *output);
extern "C" void score_RandomForest(double *input, double *output);
extern "C" void score_LogisticRegression(double *input, double *output);

// Generic wrapper for any score function
py::array_t<double> predict_model(py::array_t<double, py::array::c_style | py::array::forcecast> X,
                                  ssize_t input_dim,
                                  ssize_t output_dim,
                                  void(*score_func)(double*, double*)) {
    if (X.ndim() != 2) throw std::runtime_error("Input must be 2-D array");
    ssize_t nrows = X.shape(0);
    ssize_t ncols = X.shape(1);
    if (ncols != input_dim) throw std::runtime_error("Input column count mismatch");

    auto result = py::array_t<double>({nrows, output_dim});
    auto Xbuff = X.unchecked<2>();
    auto Rbuff = result.mutable_unchecked<2>();

    std::vector<double> inrow(input_dim);
    std::vector<double> outrow(output_dim);

    for (ssize_t i = 0; i < nrows; ++i) {
        for (ssize_t j = 0; j < input_dim; ++j) inrow[j] = Xbuff(i, j);
        score_func(inrow.data(), outrow.data());
        for (ssize_t k = 0; k < output_dim; ++k) Rbuff(i, k) = outrow[k];
    }
    return result;
}

PYBIND11_MODULE(mymodule, m) {
    m.doc() = "pybind11 wrapper for multiple m2cgen C models";

    // Expose each C model as a separate Python method
    m.def("predict_decision_tree",
          [](py::array_t<double> X, ssize_t input_dim, ssize_t output_dim) {
              return predict_model(X, input_dim, output_dim, score_DecisionTree);
          });

    m.def("predict_random_forest",
          [](py::array_t<double> X, ssize_t input_dim, ssize_t output_dim) {
              return predict_model(X, input_dim, output_dim, score_RandomForest);
          });

    m.def("predict_logistic_regression",
          [](py::array_t<double> X, ssize_t input_dim, ssize_t output_dim) {
              return predict_model(X, input_dim, output_dim, score_LogisticRegression);
          });
}
