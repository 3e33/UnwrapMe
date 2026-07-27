// Copyright (C) 2023-2026 Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "structures.h"
#include "utilities.h"

#include <xtensor/xmath.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xnorm.hpp>

#include <unordered_set>
#include <type_traits>

unordered_set<BEdgePtr> getBorderEdges(const unordered_set<BFacePtr>&);

vector3df getNormalised(const vector3df& v);

template <typename T>
unordered_set<BVertPtr> getVertsFromFaces(const T& bmFaces) {
  unordered_set<BVertPtr> verts;
  for (const auto& f : bmFaces) {
    verts.insert(f->verts.begin(), f->verts.end());
  }

  return verts;
}

template <typename T>
vector3df getVertsCentroid(const T& bmVerts) {
  double w = 1.0 / bmVerts.size();
  vector3df centre = {0, 0, 0};
  for (const auto& v : bmVerts) {
    centre += v->co * w;
  }

  return centre;
}

template <template <typename, typename...> class C, typename T, typename... E>
double getFacesArea(const C<T, E...>& bmFaces) {
  double area = 0.0;
  for (auto& f : bmFaces) {
    if constexpr (is_pointer_v<T>) {
      area += f->area;
    } else {
      area += f.area;
    }
  }

  return area;
}
