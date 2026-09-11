#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../DiffusionModel/SocialSIS.h"

namespace py = pybind11;

PYBIND11_MODULE(diffusion_model, m)
{
    py::class_<SocialSIS>(m, "SocialSIS")
        .def(py::init<std::string, float, double, int>())
        .def("get_num_nodes", &SocialSIS::get_num_nodes)
        .def("get_edges", &SocialSIS::get_edges)
        .def("influence", &SocialSIS::influence,
             py::arg("seeds"), py::arg("num_repeat"),
             py::call_guard<py::gil_scoped_release>())
        .def("influence_time", &SocialSIS::influence_time,
             py::arg("seeds"), py::arg("num_repeat"),
             py::call_guard<py::gil_scoped_release>());
}