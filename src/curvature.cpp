// Copyright (C) 2023-2026 Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#include "curvature.h"

#include "common.h"
// #include "dev.h"
// #include "fastcore.h"
#include "structures.h"
#include "utilities.h"

#include <cmath>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <set>
#include <unordered_set>

#include <xtensor/xadapt.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xnorm.hpp>

#include <boost/container_hash/hash.hpp>

// #include <omp.h>

namespace py = pybind11;

using namespace std;

namespace Curvature {
  PyObjectSet
  getLinkedVerts(const py::object& vert) {
    PyObjectSet verts;

    for (const auto& f : vert.attr("link_faces")) {
      for (const auto& v : f.attr("verts")) {
        verts.insert(v.cast<py::object>());
      }
    }
    verts.erase(vert);

    return verts;
  }

  matrix3df getRotationMatrix(const vector3df& v1,
                              const vector3df& v2) {
    vector3df w = xt::linalg::cross(v1, v2);

    double cosA = xt::linalg::vdot(v1, v2);
    double k = 1 / (1 + cosA);

    matrix3df M{{(w[0] * w[0] * k) + cosA},
                {(w[1] * w[0] * k) - w[2]},
                {(w[2] * w[0] * k) + w[1]},
                {(w[0] * w[1] * k) + w[2]},
                {(w[1] * w[1] * k) + cosA},
                {(w[2] * w[1] * k) - w[0]},
                {(w[0] * w[2] * k) - w[1]},
                {(w[1] * w[2] * k) + w[0]},
                {(w[2] * w[2] * k) + cosA}};

    return M;
  }

  pair<vector3df, vector3df>
  getOrthoBasis(const vector3df& n) { // unit vector
    vector3df u = xt::zeros<double>({3});
    if (abs(n[0]) >= abs(n[1])) {
      double q = 1 / sqrt(n[0] * n[0] +
                          n[2] * n[2]);
      u[0] = n[2] * q;
      u[1] = 0;
      u[2] = -n[0] * q;
    } else {
      double q = 1 / sqrt(n[1] * n[1] +
                          n[2] * n[2]);
      u[0] = 0;
      u[1] = n[2] * q;
      u[2] = -n[1] * q;
    }

    vector3df v = xt::linalg::cross(n, u);

    return make_pair(u, v);
  }

  double
  getTriangleArea(const vector3df& p,
                  const vector3df& p1,
                  const vector3df& p2) {
    auto&& m = xt::linalg::cross(p1 - p,
                                 p2 - p);
    double area = xt::norm_l2(m)() / 2;

    return area;
  }

  tuple<matrix2df, vector3df, vector3df>
  getFaceII(const py::object& bmFace) {
    auto [u, v] = getOrthoBasis(getVector3df(bmFace, "normal"));

    vector<double> A;
    vector<double> B;

    py::list edges = bmFace.attr("edges");
    for (const auto& edge : edges) {
      py::list verts = edge.attr("verts");
      auto&& e = (getVector3df(verts[0], "co") -
                  getVector3df(verts[1], "co"));
      auto n1 = getVector3df(verts[0], "normal");
      auto n2 = getVector3df(verts[1], "normal");

      A.insert(A.end(), {xt::linalg::vdot(e, u),
                         xt::linalg::vdot(e, v)});
      auto&& nvec = n2 - n1;
      B.insert(B.end(), {xt::linalg::vdot(nvec, u),
                         xt::linalg::vdot(nvec, v)});
    }

    xt::xarray<double> xA = xt::adapt(A, vector<size_t>{edges.size(), 2});
    xt::xarray<double> xB = xt::adapt(B, vector<size_t>{edges.size(), 2});

    matrix2df II = xt::transpose(
        get<0>(xt::linalg::lstsq(xA, xB, numeric_limits<double>::epsilon())));

    return make_tuple(II, u, v);
  }

  vector<py::object>
  getVertFaceEdges(const py::object& bmVert,
                   const py::object& bmFace) {
    vector<py::object> edges;
    for (const auto& e : bmFace.attr("edges")) {
      py::list verts = e.attr("verts");
      if (verts.contains(bmVert)) {
        edges.push_back(e.cast<py::object>());
      }
    }

    return edges;
  }

  matrix2df
  getTransformedII(const matrix2df& II,
                   const vector3df& up,
                   const vector3df& vp,
                   const vector3df& normalp,
                   const vector3df& uf,
                   const vector3df& vf,
                   const vector3df& normalf) {
    auto T = getRotationMatrix(normalp, normalf);
    auto&& up2 = xt::linalg::dot(T, up);
    auto&& vp2 = xt::linalg::dot(T, vp);

    vector2df tu = {xt::linalg::vdot(up2, uf),
                    xt::linalg::vdot(up2, vf)};
    vector2df tv = {xt::linalg::vdot(vp2, uf),
                    xt::linalg::vdot(vp2, vf)};

    auto e = xt::linalg::vdot(tu, xt::linalg::dot(II, tu));
    auto f = xt::linalg::vdot(tu, xt::linalg::dot(II, tv));
    auto g = xt::linalg::vdot(tv, xt::linalg::dot(II, tv));

    matrix2df tII {{e, f},
                   {f, g}};

    return tII;
  }

  double
  getFaceWeight(const py::object& bmFace,
                const py::object& bmVert) {
    auto edges = getVertFaceEdges(bmVert, bmFace);
    vector3df c =
      xt::adapt(bmFace.attr("calc_center_median")().cast<array<double, 3>>());
    py::list e1verts = edges[0].attr("verts");
    py::list e2verts = edges[1].attr("verts");
    auto&& p1 = (getVector3df(e1verts[1], "co") -
                 getVector3df(e1verts[0], "co")) / 2;
    auto&& p2 = (getVector3df(e2verts[1], "co") -
                 getVector3df(e2verts[0], "co")) / 2;

    auto&& p = getVector3df(bmVert, "co");
    auto a1 = getTriangleArea(p, c, p1);
    auto a2 = getTriangleArea(p, c, p2);

    auto w = a1 + a2;

    return w;
  }

  matrix2df
  getVertII(const py::object& bmVert) {
    auto vnormal = getVector3df(bmVert, "normal");

    double totalWeight = 0;
    matrix2df totalWeightedII = xt::zeros<double>({2, 2});

    for (const auto& face : bmVert.attr("link_faces")) {
      auto f = face.cast<py::object>();
      auto fnormal = getVector3df(f, "normal");

      auto w = getFaceWeight(f, bmVert);
      auto [II, uf, vf] = getFaceII(f);
      auto [up, vp] = getOrthoBasis(vnormal);
      auto tII = getTransformedII(II,
                                  up, vp, vnormal,
                                  uf, vf, fnormal);

      totalWeightedII += w * tII;
      totalWeight += w;
    }

    auto vertII = -totalWeightedII / totalWeight;

    return vertII;
  }

  double
  getMeanCurvature(const py::object& bmVert,
                   CachedFunctions& cachedFns) {
    auto II = cachedFns.getVertII(bmVert);
    auto eigens = xt::linalg::eigh(II);
    auto k1 = get<0>(eigens)[0];
    auto k2 = get<0>(eigens)[1];
    auto H = ((k1 + k2) / 2);

    return H;
  }

  double
  getMeanCurvatureCached(const py::object& bmVert,
                         MeanCurvatureCache& cache,
                         CachedFunctions& cachedFns) {
    if (cache.find(bmVert) == cache.end()) {
      cache[bmVert] = getMeanCurvature(bmVert, cachedFns);
    }

    return cache[bmVert];
  }

  double
  getAverageFaceH(const py::object& bmFace,
                  MeanCurvatureCache& cache,
                  CachedFunctions& cachedFns) {
    py::list verts = bmFace.attr("verts");
    double H = 0;
    for (const auto& v : verts) {
      H += getMeanCurvatureCached(v.cast<py::object>(),
                                  cache,
                                  cachedFns);
    }
    H /= verts.size();

    return H;
  }

  double
  getAverageFaceH(const py::object& bmFace,
                  MeanCurvatureCache& cache,
                  CachedFunctions& cachedFns,
                  const PyObjectSet& ignoreVerts) {
    py::list verts = bmFace.attr("verts");
    double H = 0;
    int n = 0;
    for (const auto& v_ : verts) {
      auto v = v_.cast<py::object>();
      if (ignoreVerts.find(v) != ignoreVerts.end()) {
        continue;
      }

      H += getMeanCurvatureCached(v, cache, cachedFns);
      n++;
    }
    H = (n > 0) ? H / n : 0;

    return H;
  }

  pair<py::list, py::list>
  getAdjacentFaces(const py::object& bmFace) {
    py::list adj;
    py::list halfedges;
    for (const auto& e : bmFace.attr("edges")) {
      for (const auto& f : e.attr("link_faces")) {
        if (e.attr("link_faces").contains(f)) {
          adj.append(f);
          halfedges.append(e);
        }
      }
    }

    return make_pair(adj, halfedges);
  }

  struct BridgeContext {
    MeanCurvatureCache cache;
    py::object initFace = py::none();
  };
  using BridgeContextPtr = shared_ptr<BridgeContext>;

  BridgeContextPtr
  getBridgeContext() {
    return make_shared<BridgeContext>();
  }

  inline double
  getCost(const double& a,
          const double& b) {
    double p = std::abs(a) > 0 ? std::abs(a - b) / std::abs(a) : 0;
    p = log1p(p);

    return p;
  }

  void
  selectByCurvature(py::object& bm,
                    BridgeContextPtr& bridgeContext,
                    const double& strength,
                    const int& sharpAngle) {
    CachedFunctions cachedFns;

    if (bridgeContext->initFace.is_none()) {
      for (const auto& f : bm.attr("faces")) {
        if (f.attr("select").cast<bool>() == true) {
          bridgeContext->initFace = f.cast<py::object>();

          break;
        }
      }
    }
    py::object selectedFace = bridgeContext->initFace;

    // clear
    for (const auto& f : bm.attr("faces")) {
      f.attr("select") = false;
    }
    selectedFace.attr("select") = true;

    // build
    auto ofvc = getAverageFaceH(selectedFace,
                                bridgeContext->cache,
                                cachedFns);
    PyObjectSet faces {selectedFace};
    PyObjectSet queue;

    auto adjFaces = getAdjacentFaces(selectedFace);
    for (size_t i = 0; i < adjFaces.first.size(); ++i) {
      auto adjf = adjFaces.first[i];
      auto adje = adjFaces.second[i];

      if (adje.attr("seam").cast<bool>() == true) {
        continue;
      }

      if (adje.attr("calc_face_angle")(0).cast<double>() >
          sharpAngle * RAD) {
        continue;
      }

      py::list verts = adje.attr("verts");
      auto v1c = getMeanCurvatureCached(verts[0], bridgeContext->cache, cachedFns);
      auto v2c = getMeanCurvatureCached(verts[1], bridgeContext->cache, cachedFns);
      if (getCost(ofvc, v1c) >= strength &&
          getCost(ofvc, v2c) >= strength) {
        continue;
      }

      queue.insert(adjf);
    }

    while (!queue.empty()) {
      py::object bmFace = queue.extract(queue.begin()).value();
      faces.insert(bmFace);

      auto adjFaces = getAdjacentFaces(bmFace);
      for (size_t i = 0; i < adjFaces.first.size(); ++i) {
        auto adjf = adjFaces.first[i];
        auto adje = adjFaces.second[i];

        if (adje.attr("seam").cast<bool>() == true) {
          continue;
        }

        if (adje.attr("calc_face_angle")(0).cast<double>() > sharpAngle * RAD) {
          continue;
        }

        py::list verts = adje.attr("verts");
        auto v1c = getMeanCurvatureCached(verts[0], bridgeContext->cache, cachedFns);
        auto v2c = getMeanCurvatureCached(verts[1], bridgeContext->cache, cachedFns);
        if (faces.find(adjf) != faces.end() ||
            (getCost(ofvc, v1c) >= strength &&
             getCost(ofvc, v2c) >= strength)) {
          continue;
        }

        queue.insert(adjf);
      }
    }

    for (const auto& f : faces) {
      f.attr("select") = true;
    }
  }

} // namespace Curvature

void pybind11SelectByCurvature(py::module& m) {
  py::class_<Curvature::BridgeContext,
             Curvature::BridgeContextPtr> (m, "UMContext");

  m.def("selectByCurvature", Curvature::selectByCurvature);
  m.def("getBridgeContext", Curvature::getBridgeContext);
}
