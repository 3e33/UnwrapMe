// Copyright (C) 2023-2026 Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#include "common.h"

#include "structures.h"

#include <xtensor/xnorm.hpp>

#include <unordered_set>

unordered_set<BEdgePtr> getBorderEdges(const unordered_set<BFacePtr>& bmFaces) {
  if (bmFaces.empty()) {
    return {};
  }

  unordered_set<BEdgePtr> edges;
  for (const auto& f : bmFaces) {
    for (const auto& e : f->edges) {
      edges.insert(e);
    }
  }

  unordered_set<BEdgePtr> border;
  for (const auto& e : edges) {
    for (const auto& f : e->link_faces) {
      if (bmFaces.find(f) == bmFaces.end()) {
        border.insert(e);

        break;
      }
    }
  }

  return border;
}

vector3df
getNormalised(const vector3df& v) {
  double l = xt::norm_l2(v)();
  if (isZero(l)) {
    return v;
  }

  return v / l;
}
