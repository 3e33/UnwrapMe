// Copyright (C) 2023-2026 Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#include "fitting.h"

#include "dev.h"
#include "structures.h"
#include "common.h"
#include "utilities.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <list>
#include <iterator>

// #include <omp.h>

#include <xtensor/xbuilder.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xshape.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor-blas/xlinalg.hpp>

using namespace std;

namespace Fitting {
  MNPair<matrix6df> getRotationalFieldMN(const vector3d& x,
                                         const vector3d& n) {
    matrix6df M = {{n[1] * n[1] * x[2] * x[2] - 2 * n[1] * n[2] * x[1] * x[2] +
                       n[2] * n[2] * x[1] * x[1],
                   -n[0] * n[1] * x[2] * x[2] + n[0] * n[2] * x[1] * x[2] +
                       n[1] * n[2] * x[0] * x[2] - n[2] * n[2] * x[0] * x[1],
                   n[0] * n[1] * x[1] * x[2] - n[0] * n[2] * x[1] * x[1] -
                       n[1] * n[1] * x[0] * x[2] + n[1] * n[2] * x[0] * x[1],
                   -n[0] * n[1] * x[2] + n[0] * n[2] * x[1],
                   -n[1] * n[1] * x[2] + n[1] * n[2] * x[1],
                   -n[1] * n[2] * x[2] + n[2] * n[2] * x[1]},
                  {-n[0] * n[1] * x[2] * x[2] + n[0] * n[2] * x[1] * x[2] +
                       n[1] * n[2] * x[0] * x[2] - n[2] * n[2] * x[0] * x[1],
                   n[0] * n[0] * x[2] * x[2] - 2 * n[0] * n[2] * x[0] * x[2] +
                       n[2] * n[2] * x[0] * x[0],
                   -n[0] * n[0] * x[1] * x[2] + n[0] * n[1] * x[0] * x[2] +
                       n[0] * n[2] * x[0] * x[1] - n[1] * n[2] * x[0] * x[0],
                   n[0] * n[0] * x[2] - n[0] * n[2] * x[0],
                   n[0] * n[1] * x[2] - n[1] * n[2] * x[0],
                   n[0] * n[2] * x[2] - n[2] * n[2] * x[0]},
                  {n[0] * n[1] * x[1] * x[2] - n[0] * n[2] * x[1] * x[1] -
                       n[1] * n[1] * x[0] * x[2] + n[1] * n[2] * x[0] * x[1],
                   -n[0] * n[0] * x[1] * x[2] + n[0] * n[1] * x[0] * x[2] +
                       n[0] * n[2] * x[0] * x[1] - n[1] * n[2] * x[0] * x[0],
                   n[0] * n[0] * x[1] * x[1] - 2 * n[0] * n[1] * x[0] * x[1] +
                       n[1] * n[1] * x[0] * x[0],
                   -n[0] * n[0] * x[1] + n[0] * n[1] * x[0],
                   -n[0] * n[1] * x[1] + n[1] * n[1] * x[0],
                   -n[0] * n[2] * x[1] + n[1] * n[2] * x[0]},
                  {-n[0] * n[1] * x[2] + n[0] * n[2] * x[1],
                   n[0] * n[0] * x[2] - n[0] * n[2] * x[0],
                   -n[0] * n[0] * x[1] + n[0] * n[1] * x[0],
                   n[0] * n[0],
                   n[0] * n[1],
                   n[0] * n[2]},
                  {-n[1] * n[1] * x[2] + n[1] * n[2] * x[1],
                   n[0] * n[1] * x[2] - n[1] * n[2] * x[0],
                   -n[0] * n[1] * x[1] + n[1] * n[1] * x[0],
                   n[0] * n[1],
                   n[1] * n[1],
                   n[1] * n[2]},
                  {-n[1] * n[2] * x[2] + n[2] * n[2] * x[1],
                   n[0] * n[2] * x[2] - n[2] * n[2] * x[0],
                   -n[0] * n[2] * x[1] + n[1] * n[2] * x[0],
                   n[0] * n[2],
                   n[1] * n[2],
                   n[2] * n[2]}};

    matrix6df N = {{n[1] * n[1] + n[2] * n[2] + x[1] * x[1] + x[2] * x[2],
                   -n[0] * n[1] - x[0] * x[1],
                   -n[0] * n[2] - x[0] * x[2],
                   0,
                   -x[2],
                   x[1]},
                  {-n[0] * n[1] - x[0] * x[1],
                   n[0] * n[0] + n[2] * n[2] + x[0] * x[0] + x[2] * x[2],
                   -n[1] * n[2] - x[1] * x[2],
                   x[2],
                   0,
                   -x[0]},
                  {-n[0] * n[2] - x[0] * x[2],
                   -n[1] * n[2] - x[1] * x[2],
                   n[0] * n[0] + n[1] * n[1] + x[0] * x[0] + x[1] * x[1],
                   -x[1],
                   x[0],
                   0},
                  {0, x[2], -x[1], 1, 0, 0},
                  {-x[2], 0, x[0], 0, 1, 0},
                  {x[1], -x[0], 0, 0, 0, 1}};

    return make_pair(move(M), move(N));
  }

  vector3df
  getSurfaceRotation(const unordered_set<BVertPtr>& bmVerts,
                     const vector3df& centre) {
    matrix6df M = xt::zeros<double>({6, 6});
    matrix6df N = xt::zeros<double>({6, 6});
    double w = 1.0 / bmVerts.size();

    for (const auto& v : bmVerts) {
      auto MN = getRotationalFieldMN(v->co - centre, v->normal);
      M += MN.first * w;
      N += MN.second * w;
    }

    auto evecs = xt::transpose(getGeneralisedEigenvectors(M, N));
    auto sortedErrors = getArgSortedTaubinErrors(evecs, M, N);

    auto minEvec = xt::row(evecs, sortedErrors[0]);
    auto r = xt::view(minEvec, xt::range(0, 3));

    if (xt::allclose(r, xt::zeros<double>({3}))) {
      minEvec = xt::row(evecs, sortedErrors[1]);
      r = xt::view(minEvec, xt::range(0, 3));
    }

    return getNormalised(r);
  }
} // namespace Fitting
