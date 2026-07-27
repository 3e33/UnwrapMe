// Copyright (C) 2023-2026 Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#include "fastcore.h"

#include "bConnector.h"
#include "dev.h"
#include "overlap_remove.h"
#include "structures.h"
#include "utilities.h"
#include "extraCuts.h"
#include "aStar.h"
#include "common.h"
#include "curvature.h"
#include "regionClustering.h"
#include "autoSeam.h"
#include "autoSplit.h"
#include "edgeWalk.h"
#include "fitting.h"

#include "optim.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xmasked_view.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xtensor_forward.hpp>
#include <xtensor/xvectorize.hpp>
#include <xtensor/xrandom.hpp>
#define FORCE_IMPORT_ARRAY
// #include "xtensor-python/pyarray.hpp"

#include <boost/container_hash/hash.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <execution>
#include <functional>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <ctime>
#include <random>

#include <omp.h>

namespace py = pybind11;

using namespace std;
using namespace std::placeholders;

const double SMALLSTEP = cbrt(numeric_limits<double>::epsilon());

template <typename T>
std::ostream& operator<<(std::ostream& os, const vector<T>& v) {
  os << "{";
  for (const auto& i : v) {
    os << " " << i << ",";
  }
  os << "}";

  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const vector<xt::xarray<double>>& v) {
  for (const auto& i : v) {
    os << i << endl;
  }

  return os;
}

inline double getFittingError(const vector4d& proxy, const vector3d& normal) {
  double error =
      (proxy[0] * normal[0] + proxy[1] * normal[1] + proxy[2] * normal[2]) -
      cos(proxy[3]);

  error *= error;

  return error;
}

//------------------------------------------------------------

double getDistanceToEdge(const vector3d& point, const BEdgePtr& edge) {
  // https://github.com/polymonster/maths/blob/master/maths.h
  auto dx = edge->verts[1]->co - edge->verts[0]->co;
  double m2 = xt::norm_sq(dx)();
  double s12 = xt::linalg::vdot(edge->verts[1]->co - point, dx) / m2;
  if (s12 < 0) {
    s12 = 0;
  } else if (s12 > 1) {
    s12 = 1;
  }

  auto v = s12 * edge->verts[0]->co + (1 - s12) * edge->verts[1]->co;

  return xt::norm_l2(point - v)();
}

double getFaceDistance(const BFacePtr& bmFace1, const BFacePtr& bmFace2) {
  auto joinEdge = getIntersection(bmFace1->edges, bmFace2->edges);
  double distance = getDistanceToEdge(bmFace1->centroid, joinEdge.at(0));
  distance += getDistanceToEdge(bmFace2->centroid, joinEdge.at(0));

  return distance;
}

double getCompactness(const ChartPtr& chart, const double& seedDistance) {
  if (chart->area == 0) {
    return 1.0;
  }

  auto compactness =
      xt::numeric_constants<double>::PI * seedDistance / chart->area;

  return compactness;
}

vector<BFacePtr> getAdjacentFaces(const BFacePtr& bmFace) {
  vector<BFacePtr> adjacent;
  for (const auto& edge : bmFace->edges) {
    if (edge->seam) {
      continue;
    }

    for (const auto& face : edge->link_faces) {
      if (face->hide) {
        continue;
      }

      if (face != bmFace) {
        adjacent.push_back(face);
      }
    }
  }

  return adjacent;
}

struct CachedFunctions {
  CachedFunction<decltype(&::getAdjacentFaces), BFacePtr>
      getAdjacentFaces{::getAdjacentFaces};
  CachedFunctionOrderless<decltype(&::getFaceDistance), BFacePtr, BFacePtr>
      getFaceDistance{::getFaceDistance};
};

double getBoundaries(const BFacePtr& bmFace,
                     const size_t& chartIndex,
                     const AllocMap& faceAllocation) {
  set<pair<BFacePtr, BEdgePtr>> adjacent;
  for (const auto& edge : bmFace->edges) {
    for (const auto& face : edge->link_faces) {
      if (face != bmFace) {
        adjacent.insert(make_pair(face, edge));
      }
    }
  }

  double outer = 0.0;
  double inner = 0.0;
  for (const auto& [face, edge] : adjacent) {
    if (faceAllocation[face->index] == chartIndex) {
      inner += edge->length;
    } else {
      outer += edge->length;
    }
  }

  if (inner == 0) {
    return 1.0;
  }

  return outer / inner;
}

BEdgePtr getConnectingEdge(const BFacePtr& bmFace1, const BFacePtr& bmFace2) {
  set<BEdgePtr> e1(bmFace1->edges.begin(), bmFace1->edges.end());
  set<BEdgePtr> e2(bmFace2->edges.begin(), bmFace2->edges.end());

  vector<BEdgePtr> connecting;
  set_intersection(e1.begin(), e1.end(),
                   e2.begin(), e2.end(),
                   back_inserter(connecting));

  if (!connecting.empty()) {
    return connecting[0];
  }

  return nullptr;
}

double getSharpness(const BEdgePtr& bmEdge) {
  double edgeAngle;
  if(bmEdge->faceAngle == std::numeric_limits<double>::infinity()) {
    edgeAngle = 0;
  } else {
    edgeAngle = bmEdge->faceAngle;
  }

  return (1 + (edgeAngle / xt::numeric_constants<double>::PI)) * 2;
}

pair<double, double> getCost(const ChartPtr& chart,
                             const BFacePtr& bmFace,
                             const BEdgePtr& connectingEdge,
                             const double& seedDistance,
                             const AllocMap& faceAllocation,
                             const double& alpha = 1.0,
                             const double& beta = 0.2,
                             const double& gamma = 0.5,
                             const double& delta = 1.5) {
  double fitting = pow(getFittingError(chart->proxy, bmFace->normal), alpha);
  double compactness = pow(getCompactness(chart, seedDistance), beta);
  double boundaries =
      pow(getBoundaries(bmFace, chart->index, faceAllocation), gamma);
  double sharpness = pow(getSharpness(connectingEdge), delta);
  double cost = fitting * compactness * boundaries * sharpness;

  return make_pair(cost, fitting);
}

set<BEdgePtr> getBorderEdges(const vector<BFacePtr>& bmFaces,
                             const AllocMap& faceAllocation,
                             const size_t& chartIndex) {
  if(bmFaces.empty()) {
    return {};
  }

  unordered_set<BEdgePtr> edges;
  for (const auto& f : bmFaces) {
    for (const auto& e : f->edges) {
      edges.insert(e);
    }
  }

  set<BEdgePtr> border;
  for (const auto& e : edges) {
    if (e->link_faces.size() == 1) {
      border.insert(e);

      continue;
    }
    
    for (const auto& f : e->link_faces) {
      if (faceAllocation[f->index] != chartIndex) {
        border.insert(e);
      }
    }
  }

  return border;
}

size_t getDifferenceCount(const AllocMap& m1, const AllocMap& m2) {
  size_t count = 0;

  #pragma omp parallel for reduction(+ : count)
  for (int i = 0; i < m1.size(); ++i) {
    if (m1[i] != m2[i]) {
      count++;
    }
  }

  return count;
}

unordered_set<BVertPtr> getVertsFromFaces(const vector<BFacePtr>& faces) {
  unordered_set<BVertPtr> verts;
  for (const auto& f : faces) {
    for (const auto& v : f->verts) {
      verts.insert(v);
    }
  }

  return verts;
}

xt::xarray<double> getCentroid(const unordered_set<BVertPtr>& verts) {
  auto coords = getVector2Xarray(verts, 3, [](const auto& v) { return v->co; });
  auto centre = xt::sum(coords, 0) / verts.size();

  return centre;
}

BFacePtr getClosestFace(const vector<BFacePtr>& faces,
                        const xt::xarray<double>& co) {
  auto face = faces.at(0);
  xt::xarray<double> v = face->centroid - co;
  auto distance = xt::norm_sq(v)();
  for (auto f = next(faces.begin()); f != faces.end(); f++) {
    v = (*f)->centroid - co;
    double newDistance = xt::norm_sq(v)();
    if (newDistance < distance) {
      distance = newDistance;
      face = *f;
    }
  }

  return face;
}

pair<BFacePtr, xt::xarray<double>>
getCentroidFace(const vector<BFacePtr>& bmFaces) {
  auto centroid = getCentroid(getVertsFromFaces(bmFaces));
  auto face = getClosestFace(bmFaces, centroid);

  return make_pair(face, centroid);
}

ChartPtr getInitChart(const BFacePtr& seedFace,
                      const size_t& index,
                      AllocMap& faceAllocation,
                      UnfitMap& unfitFaces,
                      CachedFunctions& cachedFns,
                      const double& faceAngleLimit) {
  auto initChart = make_shared<Chart>();
  initChart->index = index;
  initChart->area = seedFace->area;
  initChart->bmFaces.push_back(seedFace);
  initChart->proxy = {seedFace->normal[0],
                      seedFace->normal[1],
                      seedFace->normal[2],
                      0.0};
  faceAllocation[seedFace->index] = index;

  priority_queue<shared_ptr<Face>, vector<shared_ptr<Face>>, Face> queue;
  for (const auto& f : cachedFns.getAdjacentFaces(seedFace)) {
    auto face = make_shared<Face>();
    face->bmFace = f;
    face->seedDistance = cachedFns.getFaceDistance(seedFace, f);

    auto connectingEdge = getConnectingEdge(seedFace, f);

    auto [cost, fittingError] = getCost(initChart,
                                        f,
                                        connectingEdge,
                                        face->seedDistance,
                                        faceAllocation);
    face->cost = cost;
    face->fittingError = fittingError;

    if(connectingEdge->faceAngle > faceAngleLimit) {
      auto unfit = make_shared<UnfitFace>();
      unfit->chartIndex = initChart->index;
      unfit->face = face;
      unfitFaces.insert_or_assign(face->bmFace->index, unfit);

      continue;
    }

    queue.push(face);
  }
  initChart->queue = queue;

  return initChart;
}

void grow(ChartPtr& chart,
          AllocMap& faceAllocation,
          UnfitMap& unfitFaces,
          CachedFunctions& cachedFns,
          const double& maxError,
          const bool& useFitting,
          const double& faceAngleLimit) {
  auto seed = chart->queue.top();
  chart->queue.pop();

  if (useFitting &&
      seed->fittingError > maxError) {
    auto unfit = make_shared<UnfitFace>();
    unfit->chartIndex = chart->index;
    unfit->face = seed;
    unfitFaces.insert_or_assign(seed->bmFace->index, unfit);

    return;
  }

  if (faceAllocation.at(seed->bmFace->index) != -1) {
    return;
  }

  chart->bmFaces.push_back(seed->bmFace);
  faceAllocation[seed->bmFace->index] = chart->index;
  chart->area += seed->bmFace->area;

  vector<BFacePtr> adjacent = cachedFns.getAdjacentFaces(seed->bmFace);
  for (const auto& bf : adjacent) {
    if (faceAllocation.at(bf->index) == chart->index) {
      continue;
    }

    double seedDistance =
        cachedFns.getFaceDistance(bf, seed->bmFace) + seed->seedDistance;

    auto connectingEdge = getConnectingEdge(seed->bmFace, bf);

    auto [cost, fittingError] = getCost(chart,
                                        bf,
                                        connectingEdge,
                                        seedDistance,
                                        faceAllocation);
    auto growth = make_shared<Face>();
    growth->bmFace = bf;
    growth->seedDistance = seedDistance;
    growth->cost = cost;
    growth->fittingError = fittingError;

    if(connectingEdge->faceAngle > faceAngleLimit) {
      auto unfit = make_shared<UnfitFace>();
      unfit->chartIndex = chart->index;
      unfit->face = growth;
      unfitFaces.insert_or_assign(growth->bmFace->index, unfit);

      continue;
    }

    chart->queue.push(growth);
  }
}

void rebuildChart(ChartPtr& chart,
                  AllocMap& faceAllocation,
                  UnfitMap& unfitFaces,
                  CachedFunctions& cachedFns,
                  const double& faceAngleLimit,
                  double globalTemp = 1.0) {
  auto [centreFace, centroid] = getCentroidFace(chart->bmFaces);

  auto proxy = Optim::getProxy(chart->proxy,
                               chart->bmFaces,
                               globalTemp);

  using FittingType = tuple<double, BFacePtr, double>;
  vector<FittingType> fittingErrors;
  for (const auto& bf : chart->bmFaces) {
    fittingErrors.push_back(
        make_tuple(getFittingError(proxy, bf->normal), bf, -1.0));
  }
  stable_sort(fittingErrors.begin(),
              fittingErrors.end(),
              [](const auto& a, const auto& b) { return get<0>(a) < get<0>(b); });
  if (fittingErrors.size() > 10) {
    fittingErrors.resize(10);
    fittingErrors.shrink_to_fit();
  }
  for (auto& [error, bf, dist] : fittingErrors) {
    dist = xt::norm_sq(bf->centroid - centroid)();
  }

  auto minFaceItr = min_element(fittingErrors.begin(),
                                fittingErrors.end(),
                                [](const FittingType& a, const FittingType& b) {
                                  return get<2>(a) < get<2>(b);
                                });
  BFacePtr minFace = get<1>(*minFaceItr);

  chart->area = minFace->area;
  chart->bmFaces = {minFace};
  chart->proxy = proxy;
  faceAllocation[minFace->index] = chart->index;

  priority_queue<shared_ptr<Face>, vector<shared_ptr<Face>>, Face> queue;
  for (const auto& f : cachedFns.getAdjacentFaces(minFace)) {
    auto face = make_shared<Face>();
    face->bmFace = f;
    face->seedDistance = cachedFns.getFaceDistance(f, minFace);

    auto connectingEdge = getConnectingEdge(minFace, f);

    auto [cost, fittingError] = getCost(chart,
                                        f,
                                        connectingEdge,
                                        face->seedDistance,
                                        faceAllocation);
    face->cost = cost;
    face->fittingError = fittingError;

    if(connectingEdge->faceAngle > faceAngleLimit) {
      auto unfit = make_shared<UnfitFace>();
      unfit->chartIndex = chart->index;
      unfit->face = face;
      unfitFaces.insert_or_assign(face->bmFace->index, unfit);

      continue;
    }

    queue.push(face);
  }

  chart->queue = queue;
}

vector<size_t> getFarthestFirst(const vector<xt::xarray<double>>& points,
                                size_t startingIndex = 0,
                                size_t k = 0) {
  vector<size_t> restIndexes;
  for (size_t i = 0; i < points.size(); i++) {
    restIndexes.push_back(i);
  }

  if (k >= points.size()) {
    return restIndexes;
  }

  vector<size_t> selectedIndexes = {startingIndex};
  restIndexes.erase(
      find(restIndexes.begin(), restIndexes.end(), startingIndex));
  restIndexes.shrink_to_fit();

  vector<double> distancesCache(points.size(), std::numeric_limits<double>::infinity());

  while (selectedIndexes.size() < k) {
    vector<double> distances;
    distances.resize(restIndexes.size());

    #pragma omp parallel for
    for (int j = 0; j < restIndexes.size(); ++j) {
      size_t i = restIndexes[j];
      double tmp = xt::norm_sq(points[i] - points[selectedIndexes.back()])();

      if (tmp < distancesCache[i]) {
        distancesCache[i] = tmp;
      }

      distances[j] = distancesCache[i];
    }

    size_t maxIndex = distance(distances.begin(),
                               max_element(distances.begin(), distances.end()));
    size_t pointsIndex = restIndexes[maxIndex];

    selectedIndexes.push_back(pointsIndex);
    restIndexes.erase(
        find(restIndexes.begin(), restIndexes.end(), pointsIndex));
    restIndexes.shrink_to_fit();
  }

  return selectedIndexes;
}

void doGrowthProcedure(vector<ChartPtr>& charts,
                       AllocMap& faceAllocation,
                       UnfitMap& unfitFaces,
                       CachedFunctions& cachedFns,
                       const double& maxError,
                       const double& faceAngleLimit,
                       const bool& useFitting = true) {
  while (true) {
    vector<ChartPtr> unfinishedCharts;
    for (const auto& c : charts) {
      if (c->queue.empty()) {
        continue;
      }

      unfinishedCharts.push_back(c);
    }

    if (unfinishedCharts.empty()) {
      break;
    }

    ChartPtr minChart =
        *min_element(unfinishedCharts.begin(),
                     unfinishedCharts.end(),
                     [](const ChartPtr& a, const ChartPtr& b) {
                       return a->queue.top()->cost < b->queue.top()->cost;
                     });
    grow(minChart,
         faceAllocation,
         unfitFaces,
         cachedFns,
         maxError,
         useFitting,
         faceAngleLimit);
  }
}

template <typename T>
T getInnerSeamFaces(const BFacePtr& bmFace) {
  if (bmFace == nullptr) {
    return {};
  }

  vector<BFacePtr> queue = {bmFace};
  unordered_set<BFacePtr> done;
  done.insert(bmFace);

  while (!queue.empty()) {
    BFacePtr bf = queue.back();
    queue.pop_back();

    auto adjacent = getAdjacentFaces(bf);

    for (const auto& f : adjacent) {
      if (done.find(f) != done.end()) {
        continue;
      }

      queue.push_back(f);
      done.insert(f);
    }
  }

  return T(done.begin(), done.end());
}

vector<BFacePtr> virtualGrow(const xt::xarray<double>& proxy,
                             const set<BFacePtr>& bmFaces,
                             const size_t& chartRestriction,
                             const AllocMap& faceAllocation,
                             const double& maxError) {
  set<BFacePtr> queue = bmFaces;
  vector<BFacePtr> faces;

  set<size_t> done;
  for (const auto& f : queue) {
    done.insert(f->index);
  }

  while (!queue.empty()) {
    BFacePtr bmFace = *prev(queue.end());
    queue.erase(prev(queue.end()));

    auto fittingError = getFittingError(proxy, bmFace->normal);
    if (fittingError > maxError) {
      continue;
    }

    faces.push_back(bmFace);

    auto adjacent = getAdjacentFaces(bmFace);
    for (const auto& f : adjacent) {
      if (done.find(f->index) != done.end()) {
        continue;
      }

      if (faceAllocation.at(f->index) != chartRestriction) {
        continue;
      }

      queue.insert(f);
      done.insert(f->index);
    }
  }

  return faces;
}

void straightenBoundaries(vector<ChartPtr>& charts,
                          AllocMap& faceAllocation,
                          const BmeshPtr& bm,
                          const double& maxError,
                          const double& faceAngleLimit) {
  for (auto& c : charts) {
    c->borderEdges = getBorderEdges(c->bmFaces, faceAllocation, c->index);

    for (auto& e : c->borderEdges) {
      e->seam = true;
    }
  }

  vector<pair<size_t, size_t>> borderingCharts;
  for (const auto& combo : getCombinations(charts, 2)) {
    auto border =
        getIntersection(combo.at(0)->borderEdges, combo.at(1)->borderEdges);

    if (!border.empty()) {
      borderingCharts.push_back(
          make_pair(combo.at(0)->index, combo.at(1)->index));
    }
  }

  for (const auto& bc : borderingCharts) {
    auto borderEdges = getIntersection(charts.at(bc.first)->borderEdges,
                                       charts.at(bc.second)->borderEdges);

    if (borderEdges.empty()) {
      continue;
    }

    map<size_t, set<BFacePtr>> adjoiningFaces;
    for (const auto& e : borderEdges) {
      for (const auto& le : e->link_faces) {
        auto allocatedIndex = faceAllocation.at(le->index);
        adjoiningFaces[allocatedIndex].insert(le);
      }
    }

    auto chart1 = adjoiningFaces.begin()->first;
    auto chart2 = next(adjoiningFaces.begin())->first;

    vector<BFacePtr> overlappingFaces;
    auto vg1 = virtualGrow(charts.at(chart1)->proxy,
                           adjoiningFaces.at(chart2),
                           chart2,
                           faceAllocation,
                           maxError);
    overlappingFaces.insert(overlappingFaces.end(), vg1.begin(), vg1.end());

    auto vg2 = virtualGrow(charts.at(chart2)->proxy,
                           adjoiningFaces.at(chart1),
                           chart1,
                           faceAllocation,
                           maxError);
    overlappingFaces.insert(overlappingFaces.end(), vg2.begin(), vg2.end());

    set<BEdgePtr> overlappingFacesEdges;
    for (const auto& f : overlappingFaces) {
      for (const auto& e : f->edges) {
        overlappingFacesEdges.insert(e);
      }
    }

    set<BEdgePtr> c1b;
    set_difference(charts.at(chart1)->borderEdges.begin(),
                   charts.at(chart1)->borderEdges.end(),
                   borderEdges.begin(),
                   borderEdges.end(),
                   inserter(c1b, c1b.begin()));
    set<BEdgePtr> c2b;
    set_difference(charts.at(chart2)->borderEdges.begin(),
                   charts.at(chart2)->borderEdges.end(),
                   borderEdges.begin(),
                   borderEdges.end(),
                   inserter(c2b, c2b.begin()));
    set<BEdgePtr> totalBoundary;
    set_union(c1b.begin(), c1b.end(),
              c2b.begin(), c2b.end(),
              inserter(totalBoundary, totalBoundary.begin()));
    set<BVertPtr> boundaryVerts; 
    for (const auto& e : totalBoundary) {
      for (const auto& v : e->verts) {
        boundaryVerts.insert(v);
      }
    }
    set<BEdgePtr> unallowedEdges;
    for (const auto& v : boundaryVerts) {
      for (const auto& e : v->link_edges) {
        unallowedEdges.insert(e);
      }
    }
    set<BEdgePtr> allowedEdges;
    set_difference(overlappingFacesEdges.begin(),
                   overlappingFacesEdges.end(),
                   unallowedEdges.begin(),
                   unallowedEdges.end(),
                   inserter(allowedEdges, allowedEdges.begin()));

    for (auto it = allowedEdges.begin(); it != allowedEdges.end(); ) {
      if((*it)->faceAngle > faceAngleLimit) {
        it = allowedEdges.erase(it);

        continue;
      }

      it++;
    }


    auto intersectedBorderEdges =
        getIntersection(borderEdges, allowedEdges);

    set<BVertPtr> cutoffPoints;
    for (const auto& e : intersectedBorderEdges) {
      for (const auto& v : e->verts) {
        size_t count = 0;
        for (const auto& le : v->link_edges) {
          if (intersectedBorderEdges.find(le) != intersectedBorderEdges.end()) {
            count += 1;
          }
        }

        if (count == 1) {
          cutoffPoints.insert(v);
        }
      }
    }

    set<BEdgePtr> removedEdges;
    vector<set<BVertPtr>> endPoints;
    while (cutoffPoints.size() != 0) {
      set<BVertPtr> ends;
      BVertPtr currentVert = *prev(cutoffPoints.end());
      cutoffPoints.erase(prev(cutoffPoints.end()));
      ends.insert(currentVert);

      BEdgePtr nextEdge = nullptr;
      while (true) {
        BEdgePtr tmp = nullptr;
        for (const auto& e : currentVert->link_edges) {
          if (intersectedBorderEdges.find(e) == intersectedBorderEdges.end()) {
            continue;
          }

          if (e == nextEdge) {
            continue;
          }

          tmp = e;
        }

        nextEdge = tmp;

        if (nextEdge == nullptr) {
          break;
        }

        currentVert = getEdgeOtherVert(nextEdge, currentVert);
        nextEdge->seam = false;
        removedEdges.insert(nextEdge);
      }

      ends.insert(currentVert);
      cutoffPoints.erase(currentVert);
      endPoints.push_back(ends);
    }

    for (auto& p : endPoints) {
      if (p.empty()) {
        continue;
      }

      auto p1 = *prev(p.end());
      p.erase(prev(p.end()));
      auto p2 = *prev(p.end());
      p.erase(prev(p.end()));

      auto path = getShortestPath(p1, p2, allowedEdges);
      if (!path.empty()) {
        for (auto& e : path) {
          e->seam = true;
        }
      }
    }

    set<BEdgePtr> chart1RemainingEdges;
    set_difference(charts.at(chart1)->borderEdges.begin(),
                   charts.at(chart1)->borderEdges.end(),
                   removedEdges.begin(),
                   removedEdges.end(),
                   inserter(chart1RemainingEdges, chart1RemainingEdges.begin()));

    BFacePtr chartFace = nullptr;
    if (!chart1RemainingEdges.empty()) {
      for (const auto& f : (*chart1RemainingEdges.begin())->link_faces) {
        if (faceAllocation.at(f->index) == chart1) {
          chartFace = f;

          break;
        }
      }
    }

    charts.at(chart1)->bmFaces = getInnerSeamFaces<vector<BFacePtr>>(chartFace);
    for (const auto& f : charts.at(chart1)->bmFaces) {
      faceAllocation[f->index] = chart1;
    }
    charts.at(chart1)->borderEdges = getBorderEdges(
        charts.at(chart1)->bmFaces, faceAllocation, chart1);

    set<BEdgePtr> chart2RemainingEdges;
    set_difference(charts.at(chart2)->borderEdges.begin(),
                   charts.at(chart2)->borderEdges.end(),
                   removedEdges.begin(),
                   removedEdges.end(),
                   inserter(chart2RemainingEdges, chart2RemainingEdges.begin()));

    chartFace = nullptr;
    if (!chart2RemainingEdges.empty()) {
      for (const auto& f : (*chart2RemainingEdges.begin())->link_faces) {
        if (faceAllocation.at(f->index) == chart2) {
          chartFace = f;

          break;
        }
      }
    }

    charts.at(chart2)->bmFaces = getInnerSeamFaces<vector<BFacePtr>>(chartFace);
    for (const auto& f : charts.at(chart2)->bmFaces) {
      faceAllocation[f->index] = chart2;
    }
    charts.at(chart2)->borderEdges = getBorderEdges(
        charts.at(chart2)->bmFaces, faceAllocation, chart2);
  }

  for (auto& c : charts) {
    c->borderEdges = getBorderEdges(c->bmFaces, faceAllocation, c->index);
  }
}

UnfitMap fillHoles(vector<ChartPtr>& charts,
                   AllocMap& faceAllocation,
                   UnfitMap& unfitFaces,
                   CachedFunctions& cachedFns,
                   const int& maxIterations,
                   const double& maxError,
                   const bool& fillLargeHoles,
                   const double& faceAngleLimit,
                   double decayRate) {
  for (auto& c : charts) {
    c->borderEdges = getBorderEdges(c->bmFaces, faceAllocation, c->index);

    for (auto& e : c->borderEdges) {
      e->seam = true;
    }
  }

  vector<tuple<ChartPtr, unordered_set<BFacePtr>, BFacePtr>> holes;
  for (const auto& [ignore, uf] : unfitFaces) {
    if (faceAllocation[uf->face->bmFace->index] > -1) {
      continue;
    }

    bool found = false;
    for (const auto& [c, bFaces, s] : holes) {
      if (bFaces.find(uf->face->bmFace) != bFaces.end()) {
        found = true;

        break;
      }
    }

    if (found) {
      continue;
    }

    auto holeFaces =
        getInnerSeamFaces<unordered_set<BFacePtr>>(uf->face->bmFace);
    holes.push_back(
        make_tuple(charts.at(uf->chartIndex), holeFaces, uf->face->bmFace));
  }

  for (auto it = holes.begin(); it != holes.end();) {
    const auto& [chart, faces, seed] = *it;

    bool canMerge = false;
    auto tmp = getBorderEdges(faces);
    auto holeBorder = set<BEdgePtr>(tmp.begin(), tmp.end());
    vector<BEdgePtr> borderEdges;
    set_intersection(holeBorder.begin(), holeBorder.end(),
                     chart->borderEdges.begin(), chart->borderEdges.end(),
                     inserter(borderEdges, borderEdges.end()));
    for (const auto& e : borderEdges) {
      if(e->faceAngle <= faceAngleLimit) {
        canMerge = true;

        break;
      }
    }

    if(!canMerge) {
      it++;

      continue;
    }

    vector<BFace*> borderFaces;
    for (const auto& e : holeBorder) {
      borderFaces.insert(borderFaces.end(),
                         e->link_faces.begin(),
                         e->link_faces.end());
    }
    vector3df proxyGuess = {0.5, 0.5, 0.5};
    auto proxy = Optim::getProxy(proxyGuess,
                                 0.0,
                                 borderFaces);
    double avgError = 0.0;
    for (const auto& f : faces) {
      avgError += getFittingError(proxy, f->normal);
    }
    if (!faces.empty()) {
      avgError /= faces.size();
    }

    double holeArea = getFacesArea(faces);
    bool smallHole = false;
    if (!isZero(chart->area)) {
      smallHole = holeArea / chart->area < 0.1;
    }

    if (smallHole &&
        avgError < maxError) {
      for (const auto& f : faces) {
        chart->bmFaces.push_back(f);
        faceAllocation[f->index] = chart->index;
      }

      it = holes.erase(it);

      continue;
    }

    it++;
    }

  for (auto& c : charts) {
    for (auto& e : c->borderEdges) {
      e->seam = false;
    }
    
    c->borderEdges = getBorderEdges(c->bmFaces, faceAllocation, c->index);

    for (auto& e : c->borderEdges) {
      e->seam = true;
    }
  }

  holes.shrink_to_fit();
  unfitFaces.clear();

  if (!fillLargeHoles ||
      holes.empty()) {
    return unfitFaces;
  }

  int newChartsStartIndex = charts.size();
  for (const auto& [chart, faces, seed] : holes) {
    ChartPtr newChart = getInitChart(seed,
                                     charts.size(),
                                     faceAllocation,
                                     unfitFaces,
                                     cachedFns,
                                     faceAngleLimit);
    charts.push_back(newChart);
  }

  cout << "-----------------------------------" << endl;
  cout << "Filling disqualified areas with extra charts" << endl;

  AllocMap prevFaceAllocation = faceAllocation;
  int i = 0;
  double globalTemp = 1.0;
  for (; i <= maxIterations; i++) {
    doGrowthProcedure(charts,
                      faceAllocation,
                      unfitFaces,
                      cachedFns,
                      maxError,
                      faceAngleLimit);

    auto count = getDifferenceCount(prevFaceAllocation, faceAllocation);

    if (i >= maxIterations - 1) {
      break;
    }

    int chartFacesSum = 0;
    for (int i = newChartsStartIndex; i < charts.size(); i++) {
      chartFacesSum += charts[i]->bmFaces.size();
    }
    if (count < chartFacesSum * 0.06) {
      break;
    }

    prevFaceAllocation = faceAllocation;
    for (const auto& [chart, faces, seed] : holes) {
      for (const auto& f : faces) {
        faceAllocation[f->index] = -1;
      }
    }
    unfitFaces.clear();

    globalTemp = globalTemp * std::pow(decayRate, i);
    for_each(execution::par,
             begin(charts) + newChartsStartIndex,
             end(charts),
             [&](auto& c) {
               rebuildChart(c,
                            faceAllocation,
                            unfitFaces,
                            cachedFns,
                            faceAngleLimit,
                            globalTemp);
             });
  }

  cout << "Total extra iterations: " << i << endl;

  return unfitFaces;
}

unordered_set<ChartPtr> getAdjacentCharts(const ChartPtr& chart,
                                          const vector<ChartPtr>& charts,
                                          const AllocMap& faceAllocation) {
  unordered_set<BFacePtr> borderOuterFaces;
  for (const auto& e : chart->borderEdges) {
    for (const auto& le : e->link_faces) {
      if(faceAllocation[le->index] > -1 &&
         faceAllocation[le->index] != chart->index) {
        borderOuterFaces.insert(le);
      }
    }
  }

  unordered_set<ChartPtr> adjacent;
  for (const auto& f : borderOuterFaces) {
    adjacent.insert(charts.at(faceAllocation[f->index]));
  }

  return adjacent;
}

pair<double, double> getMergeCosts(const ChartPtr& c1, const ChartPtr& c2) {
  vector3d v1 = xt::view(c1->proxy, xt::range(0, 3));
  vector3d v2 = xt::view(c2->proxy, xt::range(0, 3));
  double vd = xt::linalg::vdot(v1, v2) / (xt::norm_l2(v1) * xt::norm_l2(v2))();
  vd = abs(1 - vd);

  double td = c1->proxy(3) / xt::numeric_constants<double>::PI -
              c2->proxy(3) / xt::numeric_constants<double>::PI;
  td = abs(td);

  return make_pair(vd, td);
}

void autoMerge(vector<ChartPtr>& charts,
               AllocMap& faceAllocation,
               const double& maxError) {
  while (true) {
    vector<ChartPtr> merge;

    for (const auto& c : charts) {
      vector<pair<pair<double, double>, ChartPtr>> costs;

      auto adjacent = getAdjacentCharts(c, charts, faceAllocation);
      for (const auto& adj : adjacent) {
        auto mergeCosts = getMergeCosts(c, adj);
        if (mergeCosts.first < 0.3 && mergeCosts.second < 0.2) {
          costs.push_back(make_pair(mergeCosts, adj));
        }
      }

      if (costs.empty()) {
        continue;
      }

      sort(costs.begin(), costs.end(),
           [](const auto& a, const auto& b) {
             return a.first < b.first;
           });
      merge.push_back(c);
      merge.push_back(costs[0].second);

      break;
    }

    if(merge.empty()) {
      break;
    }

    vector<BFacePtr> combinedFaces(merge[0]->bmFaces);
    combinedFaces.insert(combinedFaces.end(),
                         merge[0]->bmFaces.begin(),
                         merge[0]->bmFaces.end());

    vector4df proxyGuess{0.5, 0.5, 0.5, xt::numeric_constants<double>::PI_4};
    auto proxy = Optim::getProxy(proxyGuess, combinedFaces);

    if(proxy(3) > xt::numeric_constants<double>::PI_2) {
      break;
    }

    for (const auto& e : getIntersection(merge[1]->borderEdges,
                                         merge[0]->borderEdges)) {
      e->seam = false;
    }

    for (const auto& f : merge[1]->bmFaces) {
      merge[0]->bmFaces.push_back(f);
    }
    charts.erase(find(charts.begin(), charts.end(), merge[1]));
    charts.shrink_to_fit();

    for(int i = 0; i < charts.size(); i++) {
      charts[i]->index = i;
    }

    for (const auto& c : charts) {
      for (const auto& f : c->bmFaces) {
        faceAllocation[f->index] = c->index;
      }
    }

    merge[0]->borderEdges =
        getBorderEdges(merge[0]->bmFaces, faceAllocation, merge[0]->index);

    merge[0]->proxy = Optim::getProxy(merge[0]->proxy, merge[0]->bmFaces);
  }
}

vector<set<BFacePtr>> getFaceClusters(const vector<BFacePtr>& bmFaces,
                                      CachedFunctions& cachedFns) {
  vector<set<BFacePtr>> clusters;
  unordered_set<BFacePtr> faceSet(bmFaces.begin(), bmFaces.end());

  while (!faceSet.empty()) {
    set<BFacePtr> cluster;

    BFacePtr bmFace = faceSet.extract(faceSet.begin()).value();
    set<BFacePtr> queue;
    queue.insert(bmFace);

    while (!queue.empty()) {
      bmFace = *prev(queue.end());
      queue.erase(prev(queue.end()));

      cluster.insert(bmFace);
      faceSet.erase(bmFace);

      auto adjacent = cachedFns.getAdjacentFaces(bmFace);
      for (const auto& f : adjacent) {
        if (faceSet.find(f) == faceSet.end()) {
          continue;
        }

        if (cluster.find(f) != cluster.end()) {
          continue;
        }

        queue.insert(f);
      }
    }

    clusters.push_back(cluster);
  }

  return clusters;
}

ChartPtr getInitChartFromCluster(const set<BFacePtr>& seedCluster,
                                 const size_t& index,
                                 AllocMap& faceAllocation,
                                 UnfitMap& unfitFaces,
                                 CachedFunctions& cachedFns,
                                 const double& faceAngleLimit) {
  vector<BFacePtr> seedFaces(seedCluster.begin(), seedCluster.end());

  auto initChart = make_shared<Chart>();
  initChart->index = index;
  initChart->bmFaces = seedFaces;

  vector4df proxyGuess{0.5, 0.5, 0.5, xt::numeric_constants<double>::PI_4};
  initChart->proxy = Optim::getProxy(proxyGuess, seedFaces);

  rebuildChart(initChart, faceAllocation, unfitFaces, cachedFns, faceAngleLimit);

  return initChart;
}

void opGrowCharts(py::object& bm, const py::object& uiOptions) {
  CachedFunctions cachedFns;

  xt::random::seed(time(nullptr));

  double faceAngleLimit =
    uiOptions.attr("sharpAngle").cast<double>() * RAD;

  cout << endl;
  cout << "===============================" << endl;
  cout << "========== Unwrap Me ==========" << endl;
  cout << "===============================" << endl << endl;

  auto allTimer = timer("Total time:");

  auto t = timer("BMesh data transfer time:");
  auto fcBmesh = getFastcoreBmesh(bm);
  t();

  vector<BFacePtr> selected;
  if (uiOptions.attr("growFromSelected").cast<bool>()) {
    for (auto& f : fcBmesh->faces) {
      if (f.select) {
        selected.push_back(&f);
      }
    }
  } else {
    BFacePtr selectedFace = nullptr;
    if (uiOptions.attr("startFromSelected").cast<bool>()) {
      for (auto& f : fcBmesh->faces) {
        if (f.select) {
          selectedFace = &f;

          break;
        }
      }
    } else {
      srand(chrono::high_resolution_clock::now().time_since_epoch().count());
      int r = rand() % fcBmesh->faces.size();
      selectedFace = &fcBmesh->faces.at(r);
    }

    vector<xt::xarray<double>> centroids;
    transform(fcBmesh->faces.begin(),
              fcBmesh->faces.end(),
              back_inserter(centroids),
              [](const auto& f) { return f.centroid; });

    if (selectedFace == nullptr) {
      throw runtime_error("Starting face not selected.");
    }

    auto farthestTimer = timer("Equidistant face picking time: ");
    auto selectedIndexes =
        getFarthestFirst(centroids,
                         selectedFace->index,
                         uiOptions.attr("chartsNumber").cast<int>());
    farthestTimer();

    transform(selectedIndexes.begin(),
              selectedIndexes.end(),
              back_inserter(selected),
              [&](auto i) { return &fcBmesh->faces.at(i); });
  }

  AllocMap faceAllocation(fcBmesh->faces.size(), -1);
  AllocMap prevFaceAllocation(fcBmesh->faces.size(), -1);
  UnfitMap unfitFaces;

  if (selected.size() == 0) {
    throw runtime_error("No faces selected.");
  }

  vector<set<BFacePtr>> clusters = getFaceClusters(selected, cachedFns);

  vector<ChartPtr> charts;
  for (size_t i = 0; i < clusters.size(); i++) {
    ChartPtr chart;
    if (clusters.at(i).size() == 1) {
      chart = getInitChart(*clusters.at(i).begin(),
                           i,
                           faceAllocation,
                           unfitFaces,
                           cachedFns,
                           faceAngleLimit);
    } else {
      chart = getInitChartFromCluster(clusters.at(i),
                                      i,
                                      faceAllocation,
                                      unfitFaces,
                                      cachedFns,
                                      faceAngleLimit);
    }
    charts.push_back(chart);
  }

  double decayRate = 0.7;
  double globalTemp = 1.0;
  for (int i = 0; i <= uiOptions.attr("maxIterations").cast<int>(); i++) {
    cout << "Iteration: " << i << endl;
    doGrowthProcedure(charts,
                      faceAllocation,
                      unfitFaces,
                      cachedFns,
                      uiOptions.attr("maxError").cast<double>(),
                      faceAngleLimit);

    if (i >= uiOptions.attr("maxIterations").cast<int>() - 1) {
      break;
    }

    auto count = getDifferenceCount(prevFaceAllocation, faceAllocation);
    cout << "Undecided faces " << count << endl;

    int chartFacesSum = 0;
    for (const auto& fs : charts) {
      chartFacesSum += fs->bmFaces.size();
    }
    if (count < chartFacesSum * 0.06) {
      break;
    }

    prevFaceAllocation = faceAllocation;
    fill(faceAllocation.begin(), faceAllocation.end(), -1);
    unfitFaces.clear();

    globalTemp = globalTemp * std::pow(decayRate, i);
    for_each(execution::par, begin(charts), end(charts), [&](auto& c) {
      rebuildChart(
          c,
          faceAllocation,
          unfitFaces,
          cachedFns,
          faceAngleLimit,
          globalTemp);
    });
  }

  int initChartNumber = charts.size();

  while (!unfitFaces.empty()) {
    unfitFaces = fillHoles(charts,
                           faceAllocation,
                           unfitFaces,
                           cachedFns,
                           uiOptions.attr("maxIterations").cast<int>(),
                           uiOptions.attr("maxError").cast<double>(),
                           uiOptions.attr("fillHoles").cast<bool>(),
                           faceAngleLimit,
                           decayRate);
  }

  cout << "----------------------" << endl;
  cout << "Hole filling complete." << endl;
  if (uiOptions.attr("fillHoles").cast<bool>()) {
    cout << "Extra charts created: " << charts.size() - initChartNumber << endl;
  } else {
    cout << "No extra charts created." << endl;
  }

  // if (uiOptions.attr("autoMerge").cast<bool>()) {
  //   cout << "----------------------" << endl;
  //   cout << "Merging Charts:" << endl;
  //   auto mergeTimer = timer("Time: ");
  //   autoMerge(
  //       charts, faceAllocation, uiOptions.attr("maxError").cast<double>());
  //   mergeTimer();
  // }

  straightenBoundaries(charts,
                       faceAllocation,
                       fcBmesh,
                       uiOptions.attr("maxError").cast<double>(),
                       faceAngleLimit);

  if (uiOptions.attr("extraCuts").cast<bool>()) {
    makeExtraCuts(charts);
  }

  updateBmesh(bm, fcBmesh);

  cout << "==================================" << endl;
  allTimer();

  cout << endl;
}

PYBIND11_MODULE(fastcore, m) {
  m.def("growCharts", opGrowCharts);

  pybind11OverlapRemove(m);
  pybind11SelectByCurvature(m);
  pybind11RegionClustering(m);
  pybind11AutoSeam(m);
  pybind11AutoSplit(m);
  pybind11EdgeWalk(m);
}
