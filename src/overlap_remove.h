// Copyright (C) 2023-2026 Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "structures.h"

#include <pybind11/pybind11.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>

#include <set>
#include <unordered_set>
#include <vector>

namespace bg = boost::geometry;

namespace py = pybind11;

using namespace std;

using point = bg::model::point<double, 2, bg::cs::cartesian>;
using poly = bg::model::polygon<point, false>;
using polyList = vector<poly>;


BmeshPtr getMeshData(const py::object& bmesh);

vector<unordered_set<BFacePtr>> getFaceClusters(const unordered_set<BFacePtr>& bmFaces);

unordered_set<BFacePtr> getOverlappingFaces(const BmeshPtr& meshData,
                                            const polyList& uvPolygons);

vector<size_t> getOverlappingClusters(py::object& bm,
                                      const polyList& uvPolygons,
                                      py::object&);

void pybind11OverlapRemove(py::module& m);
