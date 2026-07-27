// Copyright (C) 2023-2026 Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "structures.h"

#include <unordered_set>
#include <set>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace std;

vector<set<BFacePtr>> getFaceClusters(const vector<BFacePtr>& bmFaces);

ChartPtr getInitChart(const BFacePtr& seedFace,
                      const size_t& index,
                      AllocMap& faceAllocation,
                      UnfitMap& unfitFaces,
                      const double&);

ChartPtr getInitChartFromCluster(const set<BFacePtr>& seedCluster,
                                 const size_t& index,
                                 AllocMap& faceAllocation,
                                 UnfitMap& unfitFaces,
                                 const double&);

vector<BFacePtr> getAdjacentFaces(const BFacePtr& bmFace);
