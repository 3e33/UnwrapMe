// Copyright (C) 2023-2026 Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "structures.h"

#include <pybind11/pybind11.h>

#include <set>
#include <unordered_set>
#include <vector>

void pybind11AutoSeam(pybind11::module& m);

namespace AutoSeam {
  vector<pair<BFacePtr, BEdgePtr>>
  getAdjacentFaces(const BFacePtr& bmFace);
}
