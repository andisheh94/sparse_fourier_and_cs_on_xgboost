//===============================================
// Common procedures for python objects handling
//===============================================

#ifndef PYTHON_UTILS_H
#define PYTHON_UTILS_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <vector>
#include <string>


/**
 * Starts the Python interpreter with the given
 * additional paths in PYTHONPATH.
 */
int start_python(std::vector<std::string> paths);


/**
 * Fetches a Python class in the given module.
 */
PyObject *load_class(const char *class_name, const char *module_name);

#endif
