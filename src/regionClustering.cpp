// Copyright (C) 2023-2026 Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#include "regionClustering.h"

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

namespace RegionClustering {
  struct BridgeContext;
  struct Region;
  struct DualEdge;

  using BridgeContextPtr = shared_ptr<BridgeContext>;
  using RegionPtr = unique_ptr<Region>;
  using DualEdgePtr = unique_ptr<DualEdge>;
  using FaceEdgePair = pair<py::object, py::object>;
  using FaceAllocations = PyObjectMap<Region*>;
  using EdgeAllocations = PyObjectMap<DualEdge*>;

  struct DualEdgePtrHash {
    std::size_t operator()(const DualEdgePtr& ptr) const;
  };

  struct DualEdgePtrEq {
    bool operator()(const DualEdgePtr& a, const DualEdgePtr& b) const;
  };

  struct BridgeContext {
    vector<RegionPtr> regions;
    unordered_set<DualEdgePtr, DualEdgePtrHash, DualEdgePtrEq> dualGraph;
    FaceAllocations faceAllocations;
    EdgeAllocations edgeAllocations;
    Curvature::MeanCurvatureCache curvatureCache;
    PyObjectSet selectedFaces;
    PyObjectSet selectedEdges;
    PyObjectSet selectionBoundary;
    PyObjectSet selectionBoundaryVerts;
  };

  struct Region {
    PyObjectSet bmFaces;
    PyObjectSet border;
    unordered_set<Region*> adjacencies;
    vector3df normal;
    double meanCurvature;
    xt::xtensor<double, 1> feature;
    double area;
    double perimeter;
    unordered_set<DualEdge*> outerDualEdges;

    // super region
    size_t rank = 0;
    Region* parent = this;

    unordered_set<DualEdge*> superOuterDualEdges;
    double innerEdgeCostSum = 0;
    int innerEdgeNum = 0;
    int regionNum = 1;
    double superArea = 0;
    double superPerimeter = 0;
    vector3df normalSum;
    PyObjectSet superBorder;

    void reset() {
      rank = 0;
      parent = this;

      superOuterDualEdges = outerDualEdges;
      innerEdgeCostSum = 0;
      innerEdgeNum = 0;
      regionNum = 1;
      superArea = area;
      superPerimeter = perimeter;
      normalSum = normal;
      superBorder = border;
    }
  };

  struct DualEdge {
    pair<Region*, Region*> dualEdge;
    double dihedral;
    PyObjectSet halfedges;
    double length = 0;

    double cost;
  };

  std::size_t DualEdgePtrHash::operator()(const DualEdgePtr& ptr) const {
    return boost::hash<pair<Region*, Region*>>()(ptr->dualEdge);
  };

  bool DualEdgePtrEq::operator()(const DualEdgePtr& a,
                                 const DualEdgePtr& b) const {
    return a->dualEdge == b->dualEdge;
  }

  BridgeContextPtr
  getBridgeContext() {
    return make_shared<BridgeContext>();
  }

  Region*
  getSuperRegion(Region* region) {
    if (region != region->parent) {
      region->parent = getSuperRegion(region->parent);
    }

    return region->parent;
  }

  unordered_set<Region*>
  getAdjacentSuperRegions(Region* superRegion) {
    if (superRegion->superOuterDualEdges.empty()) {
      return {};
    }

    unordered_set<Region*> adjRegions;
    for (const auto& de : superRegion->superOuterDualEdges) {
      auto sr1 = getSuperRegion(de->dualEdge.first);
      auto sr2 = getSuperRegion(de->dualEdge.second);

      if (sr1 == superRegion) {
        adjRegions.insert(sr2);
      } else {
        adjRegions.insert(sr1);
      }
    }

    return adjRegions;
  }

  void mergeRegions(Region* r1, Region* r2) {
    auto sr1 = getSuperRegion(r1);
    auto sr2 = getSuperRegion(r2);

    int regionNum = sr1->regionNum + sr2->regionNum;

    vector3df normalSum = sr1->normalSum + sr2->normalSum;

    unordered_set<DualEdge*> outerDualEdges;
    PyObjectSet border;
    border.insert(sr1->superBorder.begin(),
                  sr1->superBorder.end());
    border.insert(sr2->superBorder.begin(),
                  sr2->superBorder.end());

    double innerEdgeCostSum =
      sr1->innerEdgeCostSum + sr2->innerEdgeCostSum;
    int innerEdgeNum =
      sr1->innerEdgeNum + sr2->innerEdgeNum;
    double innerEdgeLength = 0;
    for (const auto& de : sr1->superOuterDualEdges) {
      if (sr2->superOuterDualEdges.find(de) ==
          sr2->superOuterDualEdges.end()) {
        outerDualEdges.insert(de);
      } else {
        innerEdgeCostSum += de->cost;
        innerEdgeNum++;
        innerEdgeLength += de->length;

        for (const auto& e : de->halfedges) {
          border.erase(e);
        }
      }
    }
    double superPerimeter =
        sr1->superPerimeter + sr2->superPerimeter - innerEdgeLength;

    for (const auto& de : sr2->superOuterDualEdges) {
      if (sr1->superOuterDualEdges.find(de) ==
          sr1->superOuterDualEdges.end()) {
        outerDualEdges.insert(de);
      }
    }

    double superArea = sr1->superArea + sr2->superArea;

    Region* root = nullptr;
    if (sr1->rank > sr2->rank) {
      sr2->parent = sr1;
      root = sr1;
    } else {
      sr1->parent = sr2;
      root = sr2;

      if (sr1->rank == sr2->rank) {
        sr2->rank++;
      }
    }

    root->superOuterDualEdges = move(outerDualEdges);
    root->innerEdgeCostSum = innerEdgeCostSum;
    root->innerEdgeNum = innerEdgeNum;
    root->regionNum = regionNum;
    root->superArea = superArea;
    root->superPerimeter = superPerimeter;
    root->normalSum = move(normalSum);
    root->superBorder = move(border);
  }

  template <typename T>
  pair<T, T> getOrderedPair(const T& a, const T& b) {
    auto p = make_pair(a, b);
    if (p.second < p.first) {
      swap(p.first, p.second);
    }

    return p;
  }

  vector<FaceEdgePair>
  getAdjacentFaces(const py::object& bmFace) {
    vector<FaceEdgePair> adjacentFaces;
    for (const auto& e : bmFace.attr("edges")) {
      for (const auto& f : e.attr("link_faces")) {
        if (f.is(bmFace)) {
          continue;
        }

        adjacentFaces.push_back(make_pair(f.cast<py::object>(),
                                          e.cast<py::object>()));
      }
    }

    return adjacentFaces;
  }

  unordered_set<Region*>
  getRegionAdjacencies(Region* region,
                       const FaceAllocations& faceAllocations,
                       const PyObjectSet& selectedFaces) {
    if (region->border.empty()) {
      return {};
    }

    unordered_set<Region*> adjRegions;
    for (const auto& e : region->border) {
      for (const auto& f_ : e.attr("link_faces")) {
        py::object f = f_.cast<py::object>();

        if (selectedFaces.find(f) == selectedFaces.end()) {
          continue;
        }

        adjRegions.insert(faceAllocations.at(f));
      }
    }
    adjRegions.erase(region);

    return adjRegions;
  }

  PyObjectSet
  getBorderEdges(const PyObjectSet& bmFaces) {
    if (bmFaces.empty()) {
      return {};
    }

    PyObjectSet doneEdges;
    PyObjectSet border;
    for (const auto& f : bmFaces) {
      for (const auto& e_ : f.attr("edges")) {
        auto e = e_.cast<py::object>();

        if (doneEdges.find(e) != doneEdges.end()) {
          continue;
        }

        doneEdges.insert(e);

        if (e.attr("link_faces").attr("__len__")().cast<int>() < 2) {
          continue;
        }
        
        for (const auto& f : e.attr("link_faces")) {
          if (bmFaces.find(f.cast<py::object>()) == bmFaces.end()) {
            border.insert(e);

            break;
          }
        }
      }
    }

    return border;
  }

  double
  getDihedralDegrees(const vector3df& a,
                     const vector3df& b) {
    double nd = (xt::norm_l2(a) * xt::norm_l2(b))();
    if (isZero(nd)) {
      return 0;
    }

    double d = abs(xt::linalg::vdot(a, b)) / nd;
    if (d > 1) {
      return 0;
    }

    return acos(d) * DEG;
  }

  inline
  double
  getScaled(const double& x,
            const double& a,
            const double& b,
            const double& minx,
            const double& maxx) {
    auto d = maxx - minx;
    if (isZero(d)) {
      return 0;
    }

    return a + ((x - minx) * (b - a) / d);
  }

  unordered_set<DualEdge*>
  getConnectingEdges(const Region* r1,
                     const Region* r2) {
    unordered_set<DualEdge*> connecting;

    for (const auto& ode : r1->superOuterDualEdges) {
      if (r2->superOuterDualEdges.find(ode) != r2->superOuterDualEdges.end()) {
        connecting.insert(ode);
      }
    }

    return connecting;
  }

  double
  getAverageDihedral(const unordered_set<DualEdge*> connectingEdges) {
    double avgDihedral = 0;
    int n = 0;
    for (const auto& ce : connectingEdges) {
      avgDihedral += ce->dihedral;
      n++;
    }
    avgDihedral /= n;

    return avgDihedral;
  }

  vector<PyObjectSet>
  getIsolatedClusters(const PyObjectSet& bmFaces) {
    vector<PyObjectSet> clusters;
    PyObjectSet faceSet(bmFaces.begin(),
                        bmFaces.end());

    while (!faceSet.empty()) {
      PyObjectSet cluster;

      PyObjectSet queue;
      queue.insert(*faceSet.begin());

      while (!queue.empty()) {
        auto f = queue.extract(queue.begin()).value();

        cluster.insert(f);
        faceSet.erase(f);

        PyObjectSet adjacent;
        for (const auto& e : f.attr("edges")) {
          if (e.attr("seam").cast<bool>() == true) {
            continue;
          }

          for (const auto& lf : e.attr("link_faces")) {
            adjacent.insert(lf.cast<py::object>());
          }
        }
        adjacent.erase(f);

        for (const auto& adj : adjacent) {
          if (faceSet.find(adj) == faceSet.end()) {
            continue;
          }

          if (cluster.find(adj) != cluster.end()) {
            continue;
          }

          queue.insert(adj);
        }
      }

      clusters.push_back(move(cluster));
    }

    return clusters;
  }

  struct EdgeWalkResult {
    py::object startVert;
    py::object endVert;
    PyObjectSet walkedVerts;
    PyObjectSet walkedEdges;
  };

  unique_ptr<EdgeWalkResult>
  edgeWalk(py::object initEdge,
           py::object initVert,
           const PyObjectSet& allowedEdges,
           const BridgeContext* context) {
    auto edgeWalkResult = make_unique<EdgeWalkResult>();
    edgeWalkResult->startVert = initVert;
    
    py::object otherVert = initEdge.attr("other_vert")(initVert);
    vector3df initVertCo = getVector3df(initVert, "co");
    vector3df otherVertCo = getVector3df(otherVert, "co");
    vector3df initVertNormal = getVector3df(otherVert, "normal");
    vector3df initVector = getNormalised(initVertCo - otherVertCo);
    vector3df initProjectedVector =
        initVector -
        xt::linalg::vdot(initVector, initVertNormal) * initVertNormal;
    double initEdgeSimilarity = 0;
    if (initEdge.attr("link_faces").attr("__len__")().cast<int>() == 2) {
      py::list lf = initEdge.attr("link_faces");
      initEdgeSimilarity = 1 - xt::linalg::vdot(getVector3df(lf[0], "normal"),
                                                getVector3df(lf[1], "normal"));
    }

    while (true) {
      py::object bestVert;
      py::object bestEdge;
      vector3df bestVertCo;
      vector3df bestVertNormal;
      vector3df bestProjectedVector;
      double bestEdgeSimilarity;

      double bestCost = numeric_limits<double>::infinity();
      for (const auto& e_ : initVert.attr("link_edges")) {
        auto e = e_.cast<py::object>();

        if (e.is(initEdge)) {
          continue;
        }

        if (allowedEdges.find(e) == allowedEdges.end()) {
          continue;
        }

        py::object nextVert = e.attr("other_vert")(initVert);

        vector3df nextVertCo = getVector3df(nextVert, "co");
        vector3df nextVector = nextVertCo - initVertCo;
        vector3df projectedVec =
            nextVector - xt::linalg::vdot(nextVector, initVertNormal) * initVertNormal;
        vector3df nextProjectedVector = getNormalised(projectedVec);
        double nextEdgeSimilarity = 0;
        if (e.attr("link_faces").attr("__len__")().cast<int>() == 2) {
          py::list lf = e.attr("link_faces");
          nextEdgeSimilarity =
              1 - xt::linalg::vdot(getVector3df(lf[0], "normal"),
                                   getVector3df(lf[1], "normal"));
        }

        double cost = 1 - xt::linalg::vdot(initProjectedVector, nextProjectedVector) +
          abs(initEdgeSimilarity - nextEdgeSimilarity);

        if (cost < bestCost) {
          bestCost = cost;
          bestEdge = move(e);
          bestVert = move(nextVert);
          bestVertCo = move(nextVertCo);
          bestProjectedVector = move(nextProjectedVector);
          bestEdgeSimilarity = move(nextEdgeSimilarity);
        }
      }

      if (static_cast<bool>(bestEdge) == false) {
        break;
      }

      bool end = false;
      for (const auto& e : bestVert.attr("link_edges")) {
        if (!e.is(initEdge) &&
            e.attr("seam").cast<bool>() == true) {
          end = true;

          break;
        }
        if (e.attr("link_faces").attr("__len__")().cast<int>() < 2) {
          end = true;

          break;
        }
      }
      if (edgeWalkResult->walkedVerts.find(bestVert) !=
          edgeWalkResult->walkedVerts.end()) {
        end = true;
      }
      if (context->selectionBoundaryVerts.find(bestVert) !=
          context->selectionBoundaryVerts.end()) {
        end = true;
      }
      if (end) {
        edgeWalkResult->walkedEdges.insert(bestEdge);
        edgeWalkResult->walkedVerts.insert(bestVert);
        edgeWalkResult->endVert = bestVert;

        break;
      }

      edgeWalkResult->walkedEdges.insert(bestEdge);
      edgeWalkResult->walkedVerts.insert(bestVert);
      edgeWalkResult->endVert = bestVert;

      initVert = move(bestVert);
      initEdge = move(bestEdge);
      initVertCo = move(bestVertCo);
      initVertNormal = getVector3df(initVert, "normal");
      initProjectedVector = move(bestProjectedVector);
      initEdgeSimilarity = move(bestEdgeSimilarity);
    }

    return edgeWalkResult;
  }

  int regionClustering(py::object& op,
                       const py::object& uiOptions) {
    Curvature::CachedFunctions cachedFns;

    auto bm = op.attr("bm").cast<py::object>();
    auto context = op.attr("bridgeContext").cast<BridgeContextPtr>();

    int angleLimit = uiOptions.attr("mergeAngleLimit").cast<int>();
    double mergeStrength = uiOptions.attr("mergeStrength").cast<double>();
    double curvatureWeight = uiOptions.attr("curvatureWeight").cast<double>();
    int postproSteps = uiOptions.attr("postprocessing").cast<int>();

    // reset
    for (const auto& r : context->regions) {
      r->reset();
    }

    for (const auto& e : context->selectedEdges) {
      e.attr("seam") = false;
    }

    // make regions
    if (context->regions.empty()) {
      for (const auto& f : bm.attr("faces")) {
        if (f.attr("select").cast<bool>()) {
          context->selectedFaces.insert(f.cast<py::object>());
          f.attr("select") = false;

          for (const auto& e_ : f.attr("edges")) {
            py::object e = e_.cast<py::object>();
            context->selectedEdges.insert(e);
            e.attr("seam") = false;
          }
        }
      }

      if (context->selectedFaces.empty()) {
        return -1;
      }

      context->selectionBoundary = getBorderEdges(context->selectedFaces);

      for (const auto& e : context->selectionBoundary) {
        for (const auto& v : e.attr("verts")) {
          context->selectionBoundaryVerts.insert(v.cast<py::object>());
        }
      }

      context->regions.reserve(context->selectedFaces.size());
      for (const auto& f : context->selectedFaces) {
        auto r = make_unique<Region>();

        r->bmFaces.insert(f.cast<py::object>());

        r->border = getBorderEdges(r->bmFaces);
        r->superBorder = r->border;

        r->normal = xt::adapt(f.attr("normal").cast<array<double, 3>>());
        r->normalSum = r->normal;

        r->area = f.attr("calc_area")().cast<double>();
        r->superArea = r->area;

        r->perimeter = f.attr("calc_perimeter")().cast<double>();
        r->superPerimeter = r->perimeter;

        context->faceAllocations[f] = r.get();

        context->regions.push_back(move(r));
      }

      for (const auto& r : context->regions) {
        r->adjacencies = getRegionAdjacencies(
            r.get(), context->faceAllocations, context->selectedFaces);

        for (const auto& adj : r->adjacencies) {
          auto de = make_unique<DualEdge>();

          for (const auto& e : r->border) {
            if (adj->border.find(e) != adj->border.end()) {
              de->halfedges.insert(e);
              de->length += e.attr("calc_length")().cast<double>();
            }
          }

          de->dualEdge = getOrderedPair(r.get(), adj);

          context->dualGraph.insert(move(de));
        }
      }

      for (const auto& e : context->dualGraph) {
        e->dualEdge.first->outerDualEdges.insert(e.get());
        e->dualEdge.second->outerDualEdges.insert(e.get());
        e->dualEdge.first->superOuterDualEdges = e->dualEdge.first->outerDualEdges;
        e->dualEdge.second->superOuterDualEdges = e->dualEdge.second->outerDualEdges;

        e->dihedral = getDihedralDegrees(e->dualEdge.first->normal,
                                         e->dualEdge.second->normal);

        context->edgeAllocations[*e->halfedges.begin()] = e.get();
      }
    }

    // costs
    if (curvatureWeight > 0) {
      PyObjectSet ignoredVerts;
      for (const auto& e : context->dualGraph) {
        if (e->dihedral > angleLimit) {
          py::list verts = e->halfedges.begin()->attr("verts");
          ignoredVerts.insert(verts[0]);
          ignoredVerts.insert(verts[1]);
        }
      }

      for (const auto& r : context->regions) {
        r->meanCurvature =
          Curvature::getAverageFaceH(*r->bmFaces.begin(),
                                     context->curvatureCache,
                                     cachedFns,
                                     ignoredVerts);
      }

      auto mmCurvature =
          minmax_element(context->regions.begin(),
                         context->regions.end(),
                         [](const auto& a, const auto& b) {
                           return a->meanCurvature < b->meanCurvature;
                         });

      for (const auto& r : context->regions) {
        vector<double> feature;
        feature.push_back(r->normal[0]);
        feature.push_back(r->normal[1]);
        feature.push_back(r->normal[2]);

        feature.push_back(getScaled(r->meanCurvature,
                                    -1,
                                    1,
                                    mmCurvature.first->get()->meanCurvature,
                                    mmCurvature.second->get()->meanCurvature));

        r->feature = xt::adapt(feature);
      }
    } else {
      for (const auto& r : context->regions) {
        vector<double> feature;
        feature.push_back(r->normal[0]);
        feature.push_back(r->normal[1]);
        feature.push_back(r->normal[2]);
        feature.push_back(0);

        r->feature = xt::adapt(feature);
      }
    }

    for (const auto& e : context->dualGraph) {
      if (isZero(e->dihedral)) {
        e->cost = 0;
      } else {
        auto n1 = xt::view(e->dualEdge.first->feature, xt::range(0, 3));
        auto& c1 = e->dualEdge.first->feature[3];
        auto n2 = xt::view(e->dualEdge.second->feature, xt::range(0, 3));
        auto& c2 = e->dualEdge.second->feature[3];
        e->cost =
          (1 - xt::linalg::vdot(n1, n2)) + curvatureWeight * abs(c1 - c2);
      }
    }

    // merge
    vector<DualEdge*> orderedDualEdges;
    orderedDualEdges.reserve(context->dualGraph.size());
    for (const auto& de : context->dualGraph) {
      orderedDualEdges.push_back(de.get());
    }

    sort(orderedDualEdges.begin(),
         orderedDualEdges.end(),
         [](const auto& a,
            const auto& b) { return a->cost < b->cost; });

    for (const auto& de : orderedDualEdges) {
      auto sr1 = getSuperRegion(de->dualEdge.first);
      auto sr2 = getSuperRegion(de->dualEdge.second);

      if (sr1 == sr2) {
        continue;
      }

      auto connectingEdges = getConnectingEdges(sr1, sr2);

      if (getAverageDihedral(connectingEdges) >
          angleLimit) {
        continue;
      }

      double connectingCost = 0;
      int connectingEdgeNum = 0;
      double connectingLength = 0;
      for (const auto& ce : connectingEdges) {
        connectingCost += ce->cost;
        connectingEdgeNum++;
        connectingLength += ce->length;
      }
      connectingCost /= connectingEdgeNum;

      double sr1InnerCost = (sr1->innerEdgeNum != 0) ? sr1->innerEdgeCostSum / sr1->innerEdgeNum : 0;
      double sr2InnerCost = (sr2->innerEdgeNum != 0) ? sr2->innerEdgeCostSum / sr2->innerEdgeNum : 0;

      double weight = connectingLength / min(sr1->superPerimeter, sr2->superPerimeter);

      double q1 = (sr1->superPerimeter + sr2->superPerimeter - connectingLength);
      double compactness = (q1 * q1) / (4 * xt::numeric_constants<double>::PI *
                                        (sr1->superArea + sr2->superArea));

      double d1 = xt::norm_l2(sr1->normalSum / sr1->regionNum)();
      d1 = max(d1, 0.5);
      double d2 = xt::norm_l2(sr2->normalSum / sr2->regionNum)();
      d2 = max(d2, 0.5);

      if (connectingCost <= min(sr1InnerCost + (mergeStrength / sr1->regionNum * d1 *
                                                weight * compactness),
                                sr2InnerCost + (mergeStrength / sr2->regionNum * d2 *
                                                weight * compactness))) {
        mergeRegions(de->dualEdge.first, de->dualEdge.second);
      }
    }

    // build
    unordered_map<Region*, vector<Region*>> clusters;
    for (const auto& r : context->regions) {
      clusters[getSuperRegion(r.get())].push_back(r.get());
    }

    for (const auto& [k, c] : clusters) {
      for (const auto& e : k->superBorder) {
        e.attr("seam") = true;
      }
    }

    // postprocessing
    if (postproSteps < 1) {
      return 0;
    }

    PyObjectSet grownOuterEdges;
    PyObjectSet grownFaces;

    for (const auto& [k, c] : clusters) {
      grownOuterEdges.insert(k->superBorder.begin(),
                             k->superBorder.end());
    }

    PyObjectSet newOuterEdges = grownOuterEdges;
    for (int i = 0; i < postproSteps; ++i) {
      PyObjectSet doneVerts;
      PyObjectSet erasedEdges;

      for (const auto& e : grownOuterEdges) {
        for (const auto& v_ : e.attr("verts")) {
          py::object v = v_.cast<py::object>();

          if (doneVerts.find(v) != doneVerts.end()) {
            continue;
          }

          doneVerts.insert(e);

          for (const auto& lf_ : v.attr("link_faces")) {
            auto lf = lf_.cast<py::object>();

            if (grownFaces.find(lf) != grownFaces.end()) {
              continue;
            }

            if (context->faceAllocations.find(lf) ==
                context->faceAllocations.end()) {
              continue;
            }

            grownFaces.insert(lf);

            for (const auto& fe_ : lf.attr("edges")) {
              auto fe = fe_.cast<py::object>();

              if (newOuterEdges.find(fe) != newOuterEdges.end()) {
                newOuterEdges.erase(fe);
                erasedEdges.insert(fe);
              } else {
                if (erasedEdges.find(fe) != erasedEdges.end()) {
                  continue;
                }

                newOuterEdges.insert(fe);
              }
            }
          }
        }
      }

      grownOuterEdges = newOuterEdges;
    }

    for (int i = 0; i < postproSteps; ++i) {
      PyObjectSet doneVerts;

      for (const auto& e : grownOuterEdges) {
        if (e.attr("link_faces").attr("__len__")().cast<int>() < 2) {
          continue;
        }

        for (const auto& v_ : e.attr("verts")) {
          py::object v = v_.cast<py::object>();

          if (doneVerts.find(v) != doneVerts.end()) {
            continue;
          }

          doneVerts.insert(e);

          for (const auto& lf_ : v.attr("link_faces")) {
            auto lf = lf_.cast<py::object>();

            if (grownFaces.find(lf) == grownFaces.end()) {
              continue;
            }

            grownFaces.erase(lf);

            for (const auto& fe_ : lf.attr("edges")) {
              auto fe = fe_.cast<py::object>();

              if (newOuterEdges.find(fe) == newOuterEdges.end()) {
                newOuterEdges.insert(fe);
              } else {
                newOuterEdges.erase(fe);
              }
            }
          }
        }
      }

      grownOuterEdges = newOuterEdges;
    }

    PyObjectSet allowedEdges;
    PyObjectSet allBorderEdges;

    auto isolatedRegions = getIsolatedClusters(grownFaces);
    for (const auto& faces : isolatedRegions) {
      if (faces.size() == 1 &&
          faces.begin()->attr("verts").attr("__len__")().cast<int>() == 3) {
        continue;
      }

      auto sr1 =
        getSuperRegion(context->faceAllocations[*faces.begin()]);
      if (sr1->regionNum == faces.size()) {
        continue;
      }

      for (const auto& f : faces) {
        for (const auto& e_ :f.attr("edges")) {
          auto e = e_.cast<py::object>();

          if (e.attr("link_faces").attr("__len__")().cast<int>() < 2) {
            continue;
          }

          allowedEdges.insert(e.cast<py::object>());
        }
      }

      auto border = getBorderEdges(faces);
      unordered_set<DualEdge*> borderDualEdges;

      if (getAverageDihedral(borderDualEdges) > angleLimit) {
        continue;
      }

      for (const auto& e : border) {
        if (context->selectionBoundary.find(e) !=
            context->selectionBoundary.end()) {
          continue;
        }

        if (e.attr("link_faces").attr("__len__")().cast<int>() < 2) {
          continue;
        }

        if (e.attr("calc_face_angle")(0).cast<double>() * DEG > angleLimit) {
          continue;
        }

        borderDualEdges.insert(context->edgeAllocations[e]);
        allBorderEdges.insert(e);
        e.attr("seam") = false;
      }
    }

    vector<pair<py::object,
                py::object>> endpoints;
    PyObjectSet doneVerts;
    for (const auto& e : allBorderEdges) {
      for (const auto& v_ : e.attr("verts")) {
        auto v = v_.cast<py::object>();

        if (doneVerts.find(v) != doneVerts.end()) {
          continue;
        }

        doneVerts.insert(v);

        int count = 0;
        py::object seamEdge;
        for (const auto& ve : v.attr("link_edges")) {
          if (ve.attr("seam").cast<bool>() == true) {
            count++;
            seamEdge = ve.cast<py::object>();
          }
        }

        if (count == 1) {
          endpoints.push_back(make_pair(seamEdge, v));
        }
      }
    }

    vector<unique_ptr<EdgeWalkResult>> edgeWalks;
    for (const auto& ep : endpoints) {
      auto edgeWalkResult =
          edgeWalk(ep.first, ep.second, allowedEdges, context.get());

      if (static_cast<bool>(edgeWalkResult->endVert) == false) {
        continue;
      }

      bool skip = false;
      for (const auto& ew : edgeWalks) {
        if (ew->endVert.is(edgeWalkResult->startVert)) {
          if (edgeWalkResult->endVert.is(ew->startVert) &&
              ew->endVert.is(edgeWalkResult->startVert)) {
            skip = true;

            break;
          }

          if (ew->walkedVerts.find(edgeWalkResult->endVert) !=
              ew->walkedVerts.end()) {
            skip = true;

            break;
          }
        }
      }

      if (!skip) {
        edgeWalks.push_back(move(edgeWalkResult));
      }
    }

    for (const auto& ew : edgeWalks) {
      for (const auto& e : ew->walkedEdges) {
        e.attr("seam") = true;
      }
    }

    return 0;
  }
} // namespace RegionClustering

void pybind11RegionClustering(py::module& m) {
  namespace RC = RegionClustering;

  py::module_ sm = m.def_submodule("clustering");

  sm.def("regionClustering", RC::regionClustering);

  py::class_<RC::BridgeContext,
             RC::BridgeContextPtr> (sm, "UMContext");
  sm.def("getBridgeContext", RC::getBridgeContext);
}
