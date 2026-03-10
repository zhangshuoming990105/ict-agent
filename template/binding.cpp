#include <pybind11/pybind11.h>
#include "binding_registry.h"

// Fixed binding.cpp - task implementations should not modify this file.

PYBIND11_MODULE(cuda_extension, m) {
    BindingRegistry::getInstance().applyBindings(m);
}
