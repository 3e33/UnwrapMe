// Copyright (C) 2023-2026 Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#include "extraCuts.h"

#include "structures.h"
#include "utilities.h"
#include "aStar.h"

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/index/rtree.hpp>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xmath.hpp>

#include <algorithm>
#include <iterator>
#include <vector>
#include <unordered_set>
#include <set>
#include <tuple>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

using namespace std;

using namespace xt::placeholders;

using point = bg::model::point<double, 3, bg::cs::cartesian>;
using pointValue = pair<point, BVertPtr>;
using pointTree = bgi::rtree<pointValue, bgi::quadratic<8>>;

unordered_set<BEdgePtr> getAdjacentEdges(const BEdgePtr& bmEdge) {
  unordered_set<BEdgePtr> adjacent;
  for (const auto& v : bmEdge->verts) {
    for (const auto& e : v->link_edges) {
      if (e != bmEdge) {
        adjacent.insert(e);
      }
    }
  }

  return adjacent;
}

vector<unordered_set<BEdgePtr>> getEdgeClusters(const set<BEdgePtr>& bmEdges) {
  vector<unordered_set<BEdgePtr>> clusters;
  unordered_set<BEdgePtr> edgeSet(bmEdges.begin(), bmEdges.end());

  while (!edgeSet.empty()) {
    unordered_set<BEdgePtr> cluster;

    BEdgePtr bmEdge = edgeSet.extract(edgeSet.begin()).value();
    set<BEdgePtr> queue;
    queue.insert(bmEdge);

    while (!queue.empty()) {
      bmEdge = *prev(queue.end());
      queue.erase(prev(queue.end()));

      cluster.insert(bmEdge);
      edgeSet.erase(bmEdge);

      auto adjacent = getAdjacentEdges(bmEdge);
      for (const auto& e : adjacent) {
        if (edgeSet.find(e) == edgeSet.end()) {
          continue;
        }

        if (cluster.find(e) != cluster.end()) {
          continue;
        }

        queue.insert(e);
      }
    }

    clusters.push_back(cluster);
  }

  return clusters;
}

vector<unordered_set<BVertPtr>>
getVertClusters(const vector<unordered_set<BEdgePtr>>& edgeClusters) {
  vector<unordered_set<BVertPtr>> vertClusters;

  for (const auto& ec : edgeClusters) {
    unordered_set<BVertPtr> vc;
    
    for (const auto& e : ec) {
      vc.insert(e->verts[0]);
      vc.insert(e->verts[1]);
    }

    vertClusters.push_back(vc);
  }

  return vertClusters;
}

pair<BVertPtr, BVertPtr> getClosestVerts(const pointTree& rtree,
                                         unordered_set<BVertPtr> vertCluster1,
                                         unordered_set<BVertPtr> vertCluster2) {
  vector<tuple<BVertPtr, BVertPtr, double>> vertPairDistances;

  for (const auto& v : vertCluster1) {
    vector<pointValue> verts;

    rtree.query(bgi::nearest(point(v->co[0], v->co[1], v->co[2]), 1) &&
                    bgi::satisfies([&](const pointValue& pv) {
                      return vertCluster2.find(pv.second) != vertCluster2.end();
                    }),
                back_inserter(verts));

    double dist = xt::norm_l2(v->co - verts[0].second->co)();

    vertPairDistances.push_back(make_tuple(v, verts[0].second, dist));
  }

  auto minDist = min_element(vertPairDistances.begin(),
                             vertPairDistances.end(),
                             [](const auto& a, const auto& b) {
                               return get<2>(a) < get<2>(b);
                             });

  return make_pair(get<0>(*minDist), get<1>(*minDist));
}

vector3d getClusterCentroid(unordered_set<BVertPtr> cluster) {
  auto coords = getVector2Xarray(cluster, 3, [](const auto& v) {return v->co; });

  return xt::mean(coords, 0);
}

set<BEdgePtr> getFaceEdges(vector<BFacePtr> faces) {
  set<BEdgePtr> edges;
  for (const auto& f : faces) {
    for (const auto& e : f->edges) {
      edges.insert(e);
    }
  }

  return edges;
}

set<BVertPtr> getFaceVerts(vector<BFacePtr> faces) {
  set<BVertPtr> verts;
  for (const auto& f : faces) {
    for (const auto& v : f->verts) {
      verts.insert(v);
    }
  }

  return verts;
}

double getCost(vector3d axis, BVertPtr v1, BVertPtr v2) {
  vector3d v = v2->co - v1->co;
  double d = xt::norm_l2(v)();
  v /= d;

  double cost = pow((1 - xt::linalg::dot(axis, v)()), 2) + pow(d, 0.5);

  return cost;
}

tuple<size_t, BVertPtr, BVertPtr>
getNextBestCluster(const pointTree& rtree,
                   const vector<unordered_set<BVertPtr>>& vertClusters,
                   const size_t& origin,
                   const unordered_set<size_t> ignore,
                   const vector3d& axis) {
  vector<tuple<BVertPtr, BVertPtr, double, size_t>> costs;

  for (size_t i = 1; i < vertClusters.size(); ++i) {
    if (origin == i ||
        ignore.find(i) != ignore.end()) {
      continue;
    }

    auto verts =
      getClosestVerts(rtree,
                      vertClusters[origin],
                      vertClusters[i]);

    costs.push_back(make_tuple(verts.first,
                               verts.second,
                               getCost(axis, verts.first, verts.second),
                               i));
  }

  sort(costs.begin(), costs.end(), [](const auto& a, const auto& b) {
    return get<2>(a) < get<2>(b);
  });

  return make_tuple(get<3>(costs[0]), get<0>(costs[0]), get<1>(costs[0]));
}

void makeExtraCuts(const vector<ChartPtr>& charts) {
  for (auto& c : charts) {
    auto edgeClusters = getEdgeClusters(c->borderEdges);

    if (edgeClusters.size() < 2) {
      continue;
    }

    auto clusterEdges = getFaceEdges(c->bmFaces);

    auto vertClusters = getVertClusters(edgeClusters);

    int chi = getFaceVerts(c->bmFaces).size() - clusterEdges.size() + c->bmFaces.size();
    if (chi == 1) {
      continue;
    }

    if (chi < 0 &&
        abs(c->proxy(3)) <= 1.57) {
      continue;
    }

    pointTree rtree;
    for (const auto& vc : vertClusters) {
      for (const auto& v : vc) {
        rtree.insert(make_pair(point(v->co[0],
                                     v->co[1],
                                     v->co[2]),
                               v));
      }
    }

    if (edgeClusters.size() == 2 &&
        abs(c->proxy(3)) > 1.57) {
      auto verts = getClosestVerts(rtree, vertClusters[0], vertClusters[1]);
      auto path = getShortestPath(verts.first, verts.second, clusterEdges);
      for (const auto& e : path) {
        e->seam = true;
      }

      continue;
    }

    if (abs(c->proxy(3)) < 0.4) {
      continue;
    }

    vector<pair<size_t, vector3d>> vertClusterCentroids;
    for (size_t i = 0; i < vertClusters.size(); ++i) {
      vertClusterCentroids.push_back(make_pair(i, getClusterCentroid(vertClusters[i])));
    }

    vector3d axis = xt::view(c->proxy, xt::range(_, 3));

    sort(vertClusterCentroids.begin(),
         vertClusterCentroids.end(),
         [&](const auto& a, const auto& b) {
           return xt::linalg::dot(axis, get<1>(a) - axis)() <
                  xt::linalg::dot(axis, get<1>(b) - axis)();
         });

    vector<unordered_set<BVertPtr>> sortedVertClusters;
    for (const auto& vcc : vertClusterCentroids) {
      sortedVertClusters.push_back(vertClusters[vcc.first]);
    }

    unordered_set<size_t> done;
    size_t abortCount = 0;
    size_t prevDoneSize = 0;
    size_t s = 0;
    while (true) {
      auto [nextIndex, vert1, vert2] = getNextBestCluster(
          rtree, sortedVertClusters, s, done, axis);

      for (const auto& e :
           getShortestPath(vert1, vert2, clusterEdges)) {
        e->seam = true;
      }

      if (abortCount > 10 ||
          done.size() >= vertClusters.size() - 2) {
        break;
      }

      if (nextIndex == vertClusters.size() - 1) {
        s = 0;
      } else {
        s = nextIndex;
        done.insert(nextIndex);
      }

      if (done.size() == prevDoneSize) {
        abortCount++;
      }

      prevDoneSize = done.size();
    }
  }
}
