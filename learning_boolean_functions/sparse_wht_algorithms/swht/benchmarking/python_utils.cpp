//================================================================
// Common procedures for python objects handling (implementation)
//================================================================


#include "python_utils.h"

#include <iostream>


/**
 * Starts the Python interpreter with the given
 * additional paths in PYTHONPATH.
 */
int start_python(std::vector<std::string> paths) {

    // Start interpreter
    Py_Initialize();

    // Load and update PYTHONPATH
    PyObject *module = PyImport_ImportModule("sys");
    if (module == nullptr) {
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyObject *str_pvalue = PyObject_Str(pvalue);
        std::cerr << PyUnicode_AsUTF8(str_pvalue) << std::endl;
        Py_DECREF(str_pvalue);
        Py_DECREF(ptype); Py_DECREF(pvalue); Py_DECREF(ptraceback);
        return 1;
    }
    PyObject *pythonpath = PyObject_GetAttrString(module, "path");
    Py_DECREF(module);
    for (auto &&path: paths) {
        PyObject *py_path = PyUnicode_FromString(path.c_str());
        PyList_Append(pythonpath, py_path);
        Py_DECREF(py_path);
    }
    Py_DECREF(pythonpath);
    return 0;
}

/**
 * Fetches a Python class in the given module.
 */
PyObject *load_class(const char *class_name, const char *module_name) {

    // Attempt to load module
    PyObject *module = PyImport_ImportModule(module_name);
    if (module == nullptr) {
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyObject *str_pvalue = PyObject_Str(pvalue);
        std::cerr << PyUnicode_AsUTF8(str_pvalue) << std::endl;
        Py_DECREF(str_pvalue);
        Py_XDECREF(ptype); Py_XDECREF(pvalue); Py_XDECREF(ptraceback);
        return NULL;
    }

    // Get class through module dictionary
    PyObject *module_dict = PyModule_GetDict(module);
    Py_DECREF(module);
    return PyDict_GetItemString(module_dict, class_name);
}
