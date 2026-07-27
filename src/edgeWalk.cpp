// Copyright (C) 2023-2026 Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#include "edgeWalk.h"

#include "dev.h"
#include "structures.h"
#include "common.h"
#include "utilities.h"
#include "curvature.h"

#include <pybind11/detail/internals.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
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
#include <limits>
#include <tuple>

#include <omp.h>

#include <xtensor/xbuilder.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xshape.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xaxis_slice_iterator.hpp>
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xrandom.hpp>

#include <boost/container_hash/hash.hpp>


namespace py = pybind11;

using namespace std;

namespace EdgeWalk {
  struct BridgeContext;

  using BridgeContextPtr = shared_ptr<BridgeContext>;

  struct BridgeContext {
    vector<py::object> selectedEdges;
    vector<pair<py::object, bool>> seamState;
  };

  BridgeContextPtr
  getBridgeContext() {
    return make_shared<BridgeContext>();
  }

  struct Options {
    double searchAngle;
  };

  inline
  void
  saveSeamState(const py::object& edge,
                const BridgeContextPtr& context) {
    context->seamState.emplace_back(edge, edge.attr("seam").cast<bool>());
  }

  double
  getDihedralAngle(const py::object& edge) {
    if (edge.attr("link_faces").attr("__len__")().cast<int>() < 2) {
      return 0;
    }

    auto faces = edge.attr("link_faces").cast<py::list>();
    vector3df f1n = getVector3df(faces[0], "normal");
    vector3df f2n = getVector3df(faces[1], "normal");

    return acos(clamp(xt::linalg::vdot(f1n, f2n), -1.0, 1.0));
  }

  py::object
  getNextEdge(const py::object& initEdge,
              const py::object& initVert,
              const py::object& nextVert,
              const Options& options) {
    vector3df initVertCo = getVector3df(initVert, "co");
    vector3df nextVertCo = getVector3df(nextVert, "co");
    vector3df initVector = nextVertCo - initVertCo;
    vector3df nextVertNormal = getVector3df(nextVert, "normal");
    vector3df initProjectedVector = getNormalised(
        initVector -
        xt::linalg::vdot(initVector, nextVertNormal) * nextVertNormal);
    double initEdgeDihedral = getDihedralAngle(initEdge);

    vector<pair<py::object, double>> edgeData;
    for (const auto& e_ : nextVert.attr("link_edges")) {
      auto e = e_.cast<py::object>();

      if (e.is(initEdge)) {
        continue;
      }

      py::object otherVert = e.attr("other_vert")(nextVert);
      vector3df otherVertCo = getVector3df(otherVert, "co");
      vector3df vector = otherVertCo - nextVertCo;
      vector3df projectedVector = getNormalised(
          vector -
          xt::linalg::vdot(vector, nextVertNormal) * nextVertNormal);

      if (xt::linalg::vdot(initProjectedVector, projectedVector) <= cos(options.searchAngle * RAD)) {
        continue;
      }

      double edgeDihedral = getDihedralAngle(e);
      edgeData.push_back(make_pair(e, abs(initEdgeDihedral - edgeDihedral)));
    }

    if (edgeData.empty()) {
      return py::none();
    }

    if (edgeData.size() == 1) {
      return get<0>(edgeData[0]);
    }

    return get<0>(*min_element(
        edgeData.begin(), edgeData.end(), [](const auto& a, const auto& b) {
          return get<1>(a) < get<1>(b);
        }));
  }

  PyObjectSet
  doEdgeWalk(py::object initEdge,
             py::object initVert,
             BridgeContextPtr& context,
             const Options& options) {
    PyObjectSet walkedEdges;
    PyObjectSet walkedVerts;
    initEdge.attr("seam") = true;
    walkedEdges.insert(initEdge);

    py::object nextVert = initEdge.attr("other_vert")(initVert);
    while (true) {
      bool stop = false;
      for (const auto& e : nextVert.attr("link_edges")) {
        if (e.is(initEdge)) {
          continue;
        }

        if (e.attr("link_faces").attr("__len__")().cast<int>() < 2) {
          stop = true;

          break;
        }

        if (e.attr("seam").cast<bool>()) {
          stop = true;

          break;
        }
      }

      if (stop) {
        break;
      }

      initEdge = getNextEdge(initEdge, initVert, nextVert, options);

      if (initEdge.is_none()) {
        break;
      }

      initVert = nextVert;
      nextVert = initEdge.attr("other_vert")(initVert);

      saveSeamState(initEdge, context);
      initEdge.attr("seam") = true;
      walkedEdges.insert(initEdge);
    }

    return walkedEdges;
  }

  void
  doEdgeWalkErase(py::object initEdge,
                  py::object initVert,
                  BridgeContextPtr context = nullptr) {
    initEdge.attr("seam") = false;

    py::object nextVert = initEdge.attr("other_vert")(initVert);
    py::object nextEdge;

    while (true) {
      bool stop = false;
      int count = 0;
      for (const auto& e : nextVert.attr("link_edges")) {
        if (e.is(initEdge)) {
          continue;
        }

        if (e.attr("link_faces").attr("__len__")().cast<int>() < 2) {
          stop = true;

          break;
        }

        if (e.attr("seam").cast<bool>()) {
          count++;
          nextEdge = e.cast<py::object>();
        }
      }

      if (stop ||
          count != 1 ||
          static_cast<bool>(nextEdge) == false) {
        break;
      }

      initEdge = nextEdge;
      if (context != nullptr) {
        saveSeamState(initEdge, context);
      }
      initEdge.attr("seam") = false;
      initVert = nextVert;
      nextVert = initEdge.attr("other_vert")(initVert);
    }
  }

  int edgeWalk(py::object& op,
               const py::object& props) {
    auto bm = op.attr("bm").cast<py::object>();
    auto context = op.attr("bridgeContext").cast<BridgeContextPtr>();

    Options options{props.attr("searchAngle").cast<double>()};

    if (context->selectedEdges.empty()) {
      for (const auto& e : bm.attr("edges")) {
        if (e.attr("select").cast<bool>()) {
          context->selectedEdges.push_back(e.cast<py::object>());
        }
      }
    }

    if (context->selectedEdges.empty()) {
      throw runtime_error("No edges selected.");
    }

    for (const auto& s : context->seamState) {
      get<0>(s).attr("seam") = get<1>(s);
    }
    context->seamState.clear();

    py::object initEdge = context->selectedEdges[0];
    py::object initVert = context->selectedEdges[0].attr("verts").cast<py::tuple>()[0];

    if (initEdge.attr("seam").cast<bool>()) {
      saveSeamState(initEdge, context);
      doEdgeWalkErase(initEdge, initVert, context);
      doEdgeWalkErase(initEdge, initEdge.attr("other_vert")(initVert), context);
    } else {
      saveSeamState(initEdge, context);
      auto we1 = doEdgeWalk(initEdge, initVert, context, options);
      auto we2 = doEdgeWalk(initEdge, initEdge.attr("other_vert")(initVert), context, options);
      PyObjectSet walkedEdges = move(we1);
      walkedEdges.insert(make_move_iterator(we2.begin()),
                         make_move_iterator(we2.end()));

      PyObjectSet walkedVerts;
      for (const auto& e : walkedEdges) {
        for (const auto& v : e.attr("verts")) {
          walkedVerts.insert(v.cast<py::object>());
        }
      }

      bool hasPole = false;
      for (const auto& v : walkedVerts) {
        int seamCount = 0;

        for (const auto& e_ : v.attr("link_edges")) {
          auto e = e_.cast<py::object>();
          if (walkedEdges.find(e) == walkedEdges.end()) {
            continue;
          }

          if (e.attr("seam").cast<bool>()) {
            seamCount++;
          }
        }

        if (seamCount > 2) {
          hasPole = true;

          break;
        }
      }

      if (hasPole) {
        for (const auto& v : walkedVerts) {
          int seamCount = 0;
          py::object leafEdge;
          for (const auto& e_ : v.attr("link_edges")) {
            auto e = e_.cast<py::object>();
            if (walkedEdges.find(e) == walkedEdges.end()) {
              continue;
            }

            if (e.attr("seam").cast<bool>()) {
              seamCount++;
              leafEdge = e;
            }
          }

          if (seamCount < 2) {
            doEdgeWalkErase(leafEdge, v);

            break;
          }
        }
      }
    }

    return 0;
  }
} // namespace EdgeWalk

void pybind11EdgeWalk(py::module& m) {
  py::module_ sm = m.def_submodule("edgeWalk");

  sm.def("edgeWalk", EdgeWalk::edgeWalk);

  py::class_<EdgeWalk::BridgeContext,
             EdgeWalk::BridgeContextPtr> (sm, "UMContext");
  sm.def("getBridgeContext", EdgeWalk::getBridgeContext);
}
