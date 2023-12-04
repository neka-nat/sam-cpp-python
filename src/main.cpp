#include "cpp/sam.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

class SamPredictor {
public:
    SamPredictor() {};

    ~SamPredictor() {};

    void load_model(const std::string& model_path) {
        params_.model = model_path;
        sam_ = sam_load_model(params_);
    }

    bool set_image(py::array_t<uint8_t> image) {
        auto buf = image.request();
        if (buf.ndim != 3)
            throw std::runtime_error("Number of dimensions must be 3");
        if (buf.shape[2] != 3)
            throw std::runtime_error("Number of channels must be 3");

        auto ptr = static_cast<uint8_t *>(buf.ptr);
        image_.nx = buf.shape[1];
        image_.ny = buf.shape[0];
        image_.data.insert(image_.data.end(), ptr, ptr + buf.shape[0] * buf.shape[1] * 3);

        return sam_compute_embd_img(image_, params_.n_threads, *sam_);
    }

    py::array_t<uint8_t> predict(py::array_t<float> point_coords) {
        auto buf = point_coords.request();
        if (buf.ndim != 1)
            throw std::runtime_error("Number of dimensions must be 1");
        if (buf.shape[0] != 2)
            throw std::runtime_error("Number of channels must be 2");
        auto ptr = static_cast<float *>(buf.ptr);
        sam_point point;
        point.x = ptr[0];
        point.y = ptr[1];
        std::vector<sam_image_u8> masks = sam_compute_masks(image_, params_.n_threads, point, *sam_);
        py::array_t<uint8_t> masks_arr;
        masks_arr.resize({static_cast<int>(masks.size()), image_.nx, image_.ny});
        auto buf_masks = masks_arr.request();
        auto ptr_masks = static_cast<uint8_t *>(buf_masks.ptr);
        for (size_t i = 0; i < masks.size(); i++) {
            for (size_t j = 0; j < image_.nx * image_.ny; j++) {
                ptr_masks[i * image_.nx * image_.ny + j] = masks[i].data[j];
            }
        }
        return masks_arr;
    }

    void reset_image() {
        image_.nx = 0;
        image_.ny = 0;
        image_.data.clear();
        sam_deinit(*sam_);
    }

private:
    std::shared_ptr<sam_state> sam_;
    sam_params params_;
    sam_image_u8 image_;
};


PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
        Python bindings for sam.cpp
        -----------------------

        .. currentmodule:: sam_cpp

        .. autosummary::
           :toctree: _generate

           SamPredictor
    )pbdoc";

    py::class_<SamPredictor>(m, "SamPredictor")
        .def(py::init<>())
        .def("load_model", &SamPredictor::load_model)
        .def("set_image", &SamPredictor::set_image)
        .def("predict", &SamPredictor::predict)
        .def("reset_image", &SamPredictor::reset_image);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
