// Copyright (C) 2023-2026 Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#include "autoSeam.h"

#include "dev.h"
#include "structures.h"
#include "common.h"
#include "utilities.h"
#include "bConnector.h"

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

namespace AutoSeam {
  vector<pair<BFacePtr, BEdgePtr>>
  getAdjacentFaces(const BFacePtr& bmFace) {
    vector<pair<BFacePtr, BEdgePtr>> adjacent;
    for (const auto& edge : bmFace->edges) {
      for (const auto& face : edge->link_faces) {
        if (face->hide) {
          continue;
        }

        if (edge->seam) {
          continue;
        }

        if (face != bmFace) {
          adjacent.emplace_back(face, edge);
        }
      }
    }

    return adjacent;
  }

  struct CachedFunctions {
    CachedFunction<decltype(&AutoSeam::getAdjacentFaces), BFacePtr>
        getAdjacentFaces{AutoSeam::getAdjacentFaces};
  };

  struct ClusterItem;

  struct Cluster {
    BFacePtr seed;
    vector3df proxy;
    vector<BFacePtr> faces;
    double area;
    priority_queue<ClusterItem> queue;
  };
  
  struct ClusterItem {
    BFacePtr face;
    double cost;
    Cluster* cluster;
  
    bool operator<(const ClusterItem& other) const {
      return cost > other.cost;
    }
  };

  inline
  double
  getError(const vector3df& proxy,
           const vector3df& normal) {
    return std::pow(xt::linalg::vdot(proxy, normal) - 1, 2);
  }

  void
  lloydStep(vector<Cluster>& clusters,
            CachedFunctions& cachedFns) {
    unordered_set<BFacePtr> allocations;
  
    for (auto& c : clusters) {
      allocations.insert(c.seed);
      c.area = c.seed->area;
      c.faces = {c.seed};
      c.queue = priority_queue<ClusterItem>();
      for (const auto& [adjf, adje] : cachedFns.getAdjacentFaces(c.seed)) {
        if (allocations.find(adjf) != allocations.end()) {
          continue;
        }

        c.queue.push(ClusterItem{adjf,
                                 getError(c.proxy, adjf->normal),
                                 &c});
      }
    }

    while (true) {
      Cluster* bestCluster = nullptr;
      double bestCost = std::numeric_limits<double>::infinity();
      for (auto& c : clusters) {
        if (c.queue.empty()) {
          continue;
        }

        auto item = c.queue.top();
        if (item.cost < bestCost) {
          bestCost = item.cost;
          bestCluster = &c;
        }
      }

      if (bestCluster == nullptr) {
        break;
      }

      ClusterItem item = bestCluster->queue.top();
      bestCluster->queue.pop();

      if (allocations.find(item.face) != allocations.end()) {
        continue;
      }

      item.cluster->faces.push_back(item.face);
      allocations.insert(item.face);

      item.cluster->area += item.face->area;
  
      for (const auto& [adjf, adje] : cachedFns.getAdjacentFaces(item.face)) {
        if (allocations.find(adjf) != allocations.end()) {
          continue;
        }
  
        bestCluster->queue.push(ClusterItem{adjf,
                                            getError(item.cluster->proxy, adjf->normal),
                                            item.cluster});
      }
    }
  }

  vector<unordered_set<BFacePtr>>
  getLooseGeometry(const BmeshPtr& fcBmesh,
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

  int autoSeam(py::object& op,
               const py::object& uiOptions) {
    cout << endl;
    cout << "===============================" << endl;
    cout << "========== Unwrap Me ==========" << endl;
    cout << "===============================" << endl << endl;

    CachedFunctions cachedFns;

    auto bm = op.attr("bm").cast<py::object>();

    struct Options {
      int maxIterations;
      double maxError;
      bool looseGeometry;
      bool manualSelect;
      bool clearSeams;
    } options{
        uiOptions.attr("iterations").cast<int>(),
        uiOptions.attr("maxError").cast<double>(),
        uiOptions.attr("looseGeometry").cast<bool>(),
        uiOptions.attr("manualSelect").cast<bool>(),
        uiOptions.attr("clearSeams").cast<bool>(),
    };

    auto timerData = timer("BMesh data transfer:");
    auto fcBmesh = getFastcoreBmesh(bm);
    timerData();

    vector<BFacePtr> selected;
    if (!options.manualSelect) {
      for (auto& e : fcBmesh->edges) {
        e.seam = false;
      }
      
      if (options.looseGeometry) {
        auto timerLoose = timer("Loose Geometry Clustering:");
        auto looseFaces = getLooseGeometry(fcBmesh, cachedFns);
        timerLoose();
        cout << "Disconnected Pieces: " << looseFaces.size() << endl;
        for (const auto& faces : looseFaces) {
          selected.push_back(*faces.begin());
        }
      } else {
        selected.push_back(&fcBmesh->faces.at(0));
      }
    } else {
      for (auto& f : fcBmesh->faces) {
        if (f.select) {
          selected.push_back(&f);
        }
      }

      if (options.clearSeams) {
        for (auto& e : fcBmesh->edges) {
          e.seam = false;
        }
      }

      if (selected.empty()) {
        throw runtime_error("No faces selected while in Manual Select mode.");
      }
    }

    auto timeClustering = timer("Surface Clustering:");
    vector<Cluster> clusters;
    for (const auto& f : selected) {
        clusters.push_back(
            Cluster{f,
                    f->normal,
                    {f},
                    0.0});
    }

    size_t totalIterations = 0;
    while (true) {
      size_t iteration = 0;
      vector<Cluster*> faceAllocations(fcBmesh->faces.size(), nullptr);
      while (true) {
        iteration++;
        totalIterations++;

        lloydStep(clusters, cachedFns);

        double count = 0;
        for (const auto& c : clusters) {
          for (const auto& f : c.faces) {
            if (faceAllocations[f->index] != &c) {
              count++;
            }
          }
        }
        if (count / fcBmesh->faces.size() < 0.01) {
          break;
        }

        if (iteration >= options.maxIterations) {
          break;
        }

        for (auto& c : clusters) {
          for (const auto& f : c.faces) {
            faceAllocations[f->index] = &c;
          }

          if (c.faces.size() == 1) {
            c.faces = {c.faces[0]};
            c.seed = c.faces[0];
            c.proxy = c.faces[0]->normal;
            c.area = 0.0;

            continue;
          }

          vector3df proxy{0,0,0};
          vector3df centroid{0,0,0};
          #pragma omp parallel
          {
            vector3df n_{0,0,0};
            vector3df centroid_{0,0,0};
            #pragma omp for
            for (int i = 0; i < c.faces.size(); ++i) {
              n_ += c.faces[i]->normal * c.faces[i]->area / c.area;
              centroid_ += c.faces[i]->centroid / c.faces.size();
            }
            #pragma omp critical
            {
              proxy += n_;
              centroid += centroid_;
            }
          }
          double l = xt::norm_l2(proxy)();
          if (!isZero(l)) {
            proxy /= l;
          } else {
            proxy = c.seed->normal;
          }

          struct Cost {
            BFacePtr face;
            double cost;
            double dist;
          };
          vector<Cost> costs(c.faces.size());

          #pragma omp for
          for (int i = 0; i < c.faces.size(); ++i) {
            costs[i].face = c.faces[i];
            costs[i].cost = getError(proxy, c.faces[i]->normal);
            costs[i].dist = xt::norm_sq(c.faces[i]->normal - centroid)();
          }

          sort(costs.begin(), costs.end(), [](const auto& a, const auto& b) {
            return a.cost < b.cost;
          });

          auto minFace = costs[0].face;

          if (costs.size() > 10) {
            costs.resize(10);
            costs.shrink_to_fit();

            minFace = (*min_element(costs.begin(),
                                    costs.end(),
                                    [](const auto& a, const auto& b) {
                                      return a.dist < b.dist;
                                    }))
                          .face;
          }

          c.faces = {minFace};
          c.seed = minFace;
          c.proxy = proxy;
          c.area = 0.0;
        }
      }

      vector<Cluster> clustersNew;
      for (const auto& c : clusters) {
        double errorAvg = 0;
        double maxError = -1;
        BFacePtr maxFace = nullptr;
        for (const auto& f : c.faces) {
          double error = getError(c.proxy, f->normal);
          errorAvg += error * f->area / c.area;

          if (f == c.seed) {
            continue;
          }

          if (error > maxError) {
            maxError = error;
            maxFace = f;
          }
        }

        if (maxFace == nullptr) {
          continue;
        }

        if (errorAvg > options.maxError) {
          clustersNew.push_back(Cluster{
              maxFace,
              maxFace->normal,
              {maxFace},
              0.0});
        }
      }

      if (clustersNew.empty()) {
        break;
      }

      clusters.insert(clusters.end(), clustersNew.begin(), clustersNew.end());
    }

    cout << "Charts: " << clusters.size() << endl;
    cout << "Total Iterations: " << totalIterations << endl;
    timeClustering();

    for (const auto& c : clusters) {
      unordered_set<BFacePtr> faces(c.faces.begin(), c.faces.end());
      for (const auto& e : getBorderEdges(faces)) {
        e->seam = true;
      }
    }

    updateBmesh(bm, fcBmesh);

    cout << endl;

    return 0;
  }
} // namespace AutoSeam

void pybind11AutoSeam(py::module& m) {
  py::module_ sm = m.def_submodule("autoSeam");

  sm.def("autoSeam", AutoSeam::autoSeam);
}
