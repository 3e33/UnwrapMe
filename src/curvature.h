// Copyright (C) 2023-2026 Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "structures.h"
#include "utilities.h"

#include <pybind11/pybind11.h>

#include <set>
#include <unordered_set>
#include <vector>
#include <type_traits>

namespace Curvature {
  using MeanCurvatureCache = PyObjectMap<double>;

  matrix2df
  getVertII(const pybind11::object& bmVert);

  struct CachedFunctions {
    CachedFunction<decltype(&Curvature::getVertII), pybind11::object> getVertII{
        Curvature::getVertII};
  };

  double
  getAverageFaceH(const pybind11::object& bmFace,
                  MeanCurvatureCache& cache,
                  CachedFunctions& cachedFns);
  double
  getAverageFaceH(const pybind11::object& bmFace,
                  MeanCurvatureCache& cache,
                  CachedFunctions& cachedFns,
                  const PyObjectSet& ignoreVerts);
}

void pybind11SelectByCurvature(pybind11::module& m);
