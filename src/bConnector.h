// Copyright (C) 2023-2026 Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "structures.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;

namespace py = pybind11;

BmeshPtr getFastcoreBmesh(const py::object& bmesh);

void updateBmesh(py::object& bm, BmeshPtr fcBmesh);
