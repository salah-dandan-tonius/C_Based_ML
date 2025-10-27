#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>

namespace py = pybind11;

// declare the C model function provided by m2cgen output
extern "C" void score(double *input, double *output);

py::array_t<double> predict(py::array_t<double, py::array::c_style | py::array::forcecast> X,
                            ssize_t input_dim,
                            ssize_t output_dim) {
    if (X.ndim() != 2) throw std::runtime_error("Input must be 2-D array");
    ssize_t nrows = X.shape(0);
    ssize_t ncols = X.shape(1);
    if (ncols != input_dim) {
        throw std::runtime_error("Input column count mismatch");
    }

    // allocate output array (nrows x output_dim)
    auto result = py::array_t<double>({nrows, output_dim});
    auto Xbuff = X.unchecked<2>();
    auto Rbuff = result.mutable_unchecked<2>();

    // temp row buffers (optional; call score using pointer into contiguous memory)
    std::vector<double> inrow(input_dim);
    std::vector<double> outrow(output_dim);

    for (ssize_t i = 0; i < nrows; ++i) {
        // copy row to contiguous double buffer
        for (ssize_t j = 0; j < input_dim; ++j) inrow[j] = Xbuff(i, j);
        // call C model
        score(inrow.data(), outrow.data());
        // copy outputs back
        for (ssize_t k = 0; k < output_dim; ++k) Rbuff(i, k) = outrow[k];
    }
    return result;
}

PYBIND11_MODULE(mymodule, m) {
    m.doc() = "pybind11 wrapper for m2cgen C model";
    m.def("predict", &predict, "Run model on a 2D numpy array",
          py::arg("X"), py::arg("input_dim"), py::arg("output_dim"));
}
