// Copyright (C) 2023-2026 Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#include "autoSplit.h"

#include "dev.h"
#include "structures.h"
#include "common.h"
#include "utilities.h"
#include "bConnector.h"
#include "optim.h"
#include "autoSeam.h"
#include "extraCuts.h"
#include "fitting.h"
#include "aStar.h"

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
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace py = pybind11;

using namespace std;

using point = bg::model::point<double, 3, bg::cs::cartesian>;
using pointValue = pair<point, BVertPtr>;
using pointTree = bgi::rtree<pointValue, bgi::quadratic<8>>;

namespace AutoSplit {
  struct CachedFunctions {
    CachedFunction<decltype(&AutoSeam::getAdjacentFaces), BFacePtr>
        getAdjacentFaces{AutoSeam::getAdjacentFaces};
  };

  vector<unordered_set<BFacePtr>>
  getSeamClusters(const BmeshPtr& fcBmesh,
                  CachedFunctions& cachedFns) {
    vector<unordered_set<BFacePtr>> clusters;

    unordered_set<BFacePtr> faces;
    faces.reserve(fcBmesh->faces.size());
    for (auto& f : fcBmesh->faces) {
      faces.insert(&f);
    }

    while (!faces.empty()) {
      unordered_set<BFacePtr> cluster;

      unordered_set<BFacePtr> queue;
      queue.insert(*faces.begin());

      while (!queue.empty()) {
        BFacePtr bmFace = queue.extract(queue.begin()).value();

        cluster.insert(bmFace);
        faces.erase(bmFace);

        for (const auto& [adjf, adje] : cachedFns.getAdjacentFaces(bmFace)) {
          if (cluster.find(adjf) != cluster.end()) {
            continue;
          }

          queue.insert(adjf);
        }
      }

      clusters.push_back(move(cluster));
    }

    return clusters;
  }

  vector<unordered_set<BEdgePtr>>
  getBorderEdgeClusters(unordered_set<BEdgePtr>&& edgeSet) {
    vector<unordered_set<BEdgePtr>> clusters;

    while (!edgeSet.empty()) {
      unordered_set<BEdgePtr> cluster;

      unordered_set<BEdgePtr> queue;
      queue.insert(*edgeSet.begin());

      while (!queue.empty()) {
        BEdgePtr bmEdge = queue.extract(queue.begin()).value();

        cluster.insert(bmEdge);
        edgeSet.erase(bmEdge);

        unordered_set<BEdgePtr> adjacent;
        for (const auto& v : bmEdge->verts) {
          for (const auto& e : v->link_edges) {
            if (e == bmEdge) {
              continue;
            }

            adjacent.insert(e);
          }
        }

        for (const auto& adj : adjacent) {
          if (cluster.find(adj) != cluster.end()) {
            continue;
          }

          if (edgeSet.find(adj) == edgeSet.end()) {
            continue;
          }

          queue.insert(adj);
        }
      }

      clusters.push_back(move(cluster));
    }

    return clusters;
  }

  unordered_set<BEdgePtr>
  getBorderEdges(const unordered_set<BFacePtr>& bmFaces) {
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
      if (e->link_faces.size() == 1) {
        border.insert(e);

        continue;
      }

      for (const auto& f : e->link_faces) {
        if (bmFaces.find(f) == bmFaces.end()) {
          border.insert(e);
  
          break;
        }
      }
    }
  
    return border;
  }

  pair<BVertPtr, BVertPtr>
  getClosestVerts(const pointTree& rtree,
                  const unordered_set<BVertPtr>& vertCluster1,
                  const unordered_set<BVertPtr>& vertCluster2) {
    vector<tuple<BVertPtr, BVertPtr, double>> vertPairDistances;
  
    for (const auto& v : vertCluster1) {
      vector<pointValue> verts;

      rtree.query(bgi::nearest(point(v->co[0], v->co[1], v->co[2]), 1) &&
                      bgi::satisfies([&](const pointValue& pv) {
                        return vertCluster2.find(pv.second) !=
                               vertCluster2.end();
                      }),
                  back_inserter(verts));

      double dist = xt::norm_sq(v->co - verts[0].second->co)();
  
      vertPairDistances.push_back(make_tuple(v, verts[0].second, dist));
    }
  
    auto minDist = min_element(vertPairDistances.begin(),
                               vertPairDistances.end(),
                               [](const auto& a, const auto& b) {
                                 return get<2>(a) < get<2>(b);
                               });
  
    return make_pair(get<0>(*minDist), get<1>(*minDist));
  }

  int autoSplit(py::object& op) {
    CachedFunctions cachedFns;

    auto bm = op.attr("bm").cast<py::object>();
    auto timerData = timer("BMesh data transfer:");
    auto fcBmesh = getFastcoreBmesh(bm);
    timerData();

    auto clusters = getSeamClusters(fcBmesh, cachedFns);

    for (const auto& c : clusters) {
      if (c.empty()) {
        continue;
      }

      vector3df avgNormal{0,0,0};
      double w = 1.0 / c.size();
      for (const auto& f : c) {
        avgNormal += f->normal * w;
      }
      if (xt::norm_l2(avgNormal)() > 0.9) {
        continue;
      }

      auto borderEdges = AutoSplit::getBorderEdges(c);
      for (const auto& f : c) {
        for (const auto& e : f->edges) {
          e->seam = false;
        }
      }
      for (const auto& e : borderEdges) {
        e->seam = true;
      }

      auto borderClusters = getBorderEdgeClusters(move(borderEdges));

      if (borderClusters.empty() ||
          borderClusters.size() < 2) {
        continue;
      }

      auto regionVerts = getVertsFromFaces(c);
      auto regionCentroid = getVertsCentroid(regionVerts);
      auto axis = Fitting::getSurfaceRotation(regionVerts, regionCentroid);

      vector<pair<unordered_set<BVertPtr>, double>>
        distances;
      for (auto& bc : borderClusters) {
        unordered_set<BVertPtr> borderVerts;
        for (const auto& e : bc) {
          for (const auto& v : e->verts) {
            borderVerts.insert(v);
          }
        }
        auto borderCentroid = getVertsCentroid(borderVerts);

        vector3df v1 = regionCentroid - borderCentroid;
        double dist = xt::norm_sq(v1 - xt::linalg::vdot(v1, axis) * axis)();

        distances.push_back(make_pair(move(borderVerts), dist));
      }

      sort(distances.begin(),
           distances.end(),
           [](const auto& a, const auto& b) { return get<1>(a) < get<1>(b); });

      pointTree rtree;
      for (const auto& v : distances.at(0).first) {
        rtree.insert(make_pair(point(v->co[0],
                                     v->co[1],
                                     v->co[2]), v));


      }
      for (const auto& v : distances.at(1).first) {
        rtree.insert(make_pair(point(v->co[0],
                                     v->co[1],
                                     v->co[2]), v));


      }

      auto verts = getClosestVerts(rtree, distances[0].first, distances[1].first);
      unordered_set<BEdgePtr> clusterEdges;
      for (const auto& f : c) {
        for (const auto& e : f->edges) {
          clusterEdges.insert(e);
        }
      }
      auto path = getShortestPath(verts.first, verts.second, clusterEdges);
      for (const auto& e : path) {
        e->seam = true;
      }
    }

    updateBmesh(bm, fcBmesh);

    cout << endl;

    return 0;
  }
} // namespace AutoSplit

void pybind11AutoSplit(py::module& m) {
  py::module_ sm = m.def_submodule("autoSplit");

  sm.def("autoSplit", AutoSplit::autoSplit);
}
