#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../inc/ngram.h"

// Module name follows torch's TORCH_EXTENSION_NAME macro (defined by
// torch.utils.cpp_extension.load(name="dartree_ngram_cpp")), so the exported
// PyInit_* symbol matches the extension name. DART upstream hardcodes
// "ngram_cpp" here and uses name="ngram_cpp" in load(); DARTree renames the
// extension to avoid clobbering DART's build cache, hence the macro form.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace ngram;
  namespace py = pybind11;
  m.doc() = "";

  py::class_<TrieNgram>(m, "TrieNgram")
      .def(py::init<size_t>(), py::arg("order"))
      .def("get_order", &TrieNgram::get_order, "")
      .def("add_conversation", &TrieNgram::add_conversation, "")
      .def("get_probability", &TrieNgram::get_probability, "")
      .def("save", &TrieNgram::save, "")
      .def_static("load", &TrieNgram::load, "")
      .def("add_all", &TrieNgram::add_all, "")
      .def("reduce", &TrieNgram::reduce, py::arg("threshold"), "");
}
