// Copyright (C) 2023-2026 Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#include "overlap_remove.h"

#include "dev.h"
#include "fastcore.h"
#include "structures.h"
#include "common.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/index/rtree.hpp>

#include <algorithm>
#include <functional>
#include <memory>
#include <set>
#include <unordered_set>

#include <omp.h>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace py = pybind11;

using namespace std;

using point = bg::model::point<double, 2, bg::cs::cartesian>;
using poly = bg::model::polygon<point, false>;
using polyList = vector<poly>;

using box = bg::model::box<point>;
using boxValue = pair<box, size_t>;

PYBIND11_MAKE_OPAQUE(poly);
PYBIND11_MAKE_OPAQUE(polyList);

BmeshPtr getMeshData(const py::object& bmesh) {
  auto meshData = make_shared<Bmesh>();
  meshData->faces.reserve(py::len(bmesh.attr("faces")));
  meshData->edges.reserve(py::len(bmesh.attr("edges")));
  meshData->verts.reserve(py::len(bmesh.attr("verts")));

  for (const auto& face : bmesh.attr("faces")) {
    auto f = BFace();
    f.index = face.attr("index").cast<size_t>();
    f.hide = face.attr("hide").cast<bool>();
    meshData->faces.push_back(f);
  }

  for (const auto& edge : bmesh.attr("edges")) {
    auto e = BEdge();
    e.index = edge.attr("index").cast<size_t>();
    e.length = edge.attr("calc_length")().cast<double>();
    e.seam = edge.attr("seam").cast<bool>();
    meshData->edges.push_back(e);
  }

  for (const auto& vert : bmesh.attr("verts")) {
    auto v = BVert();
    v.index = vert.attr("index").cast<size_t>();
    meshData->verts.push_back(v);
  }

  // links

  for (const auto& face : bmesh.attr("faces")) {
    auto i = face.attr("index").cast<size_t>();

    for (const auto& e : face.attr("edges")) {
      auto ei = e.attr("index").cast<size_t>();
      meshData->faces.at(i).edges.push_back(&meshData->edges.at(ei));
    }

    for (const auto& v : face.attr("verts")) {
      auto vi = v.attr("index").cast<size_t>();
      meshData->faces.at(i).verts.push_back(&meshData->verts.at(vi));
    }
  }

  for (const auto& edge : bmesh.attr("edges")) {
    auto i = edge.attr("index").cast<size_t>();

    for (const auto& v : edge.attr("verts")) {
      auto vi = v.attr("index").cast<size_t>();
      meshData->edges.at(i).verts.push_back(&meshData->verts.at(vi));
    }

    for (const auto& f : edge.attr("link_faces")) {
      auto fi = f.attr("index").cast<size_t>();
      meshData->edges.at(i).link_faces.push_back(&meshData->faces.at(fi));
    }
  }

  for (const auto& vert : bmesh.attr("verts")) {
    auto i = vert.attr("index").cast<size_t>();

    for (const auto& e : vert.attr("link_edges")) {
      auto ei = e.attr("index").cast<size_t>();
      meshData->verts.at(i).link_edges.push_back(&meshData->edges.at(ei));
    }
  }

  return meshData;
}

template <typename T>
bool isPolygonIntersecting(const bgi::rtree<boxValue, T>& rtree,
                           const boxValue& value,
                           const polyList& polygons,
                           const bool useActive = false,
                           const unordered_set<size_t>& activePolys = {}) {
  vector<boxValue> values;

  if (!useActive) {
    rtree.query(bgi::intersects(value.first), back_inserter(values));
  } else {
    rtree.query(bgi::intersects(value.first) &&
                    bgi::satisfies([&](const boxValue& bv) {
                      return activePolys.find(bv.second) != activePolys.end();
                    }),
                back_inserter(values));
  }

  for (const auto& [b, index] : values) {
    if (index == value.second) {
      continue;
    }

    if (bg::overlaps(polygons[value.second], polygons[index]) ||
        bg::covered_by(polygons[value.second], polygons[index]) ||
        bg::covered_by(polygons[index], polygons[value.second])) {
      return true;
    }
  }

  return false;
}

template <typename T>
std::unordered_set<BFacePtr>
getOverlappingFaces(const BmeshPtr& meshData,
                    const polyList& uvPolygons,
                    const vector<boxValue>& rtreeValues,
                    const bgi::rtree<boxValue, T>& rtree) {
  unordered_set<BFacePtr> overlappingFaces;

  #pragma omp parallel for
  for (int i = 0; i < rtreeValues.size(); ++i) {
    if (meshData->faces[i].hide) {
      continue;
    }

    if (isPolygonIntersecting(rtree, rtreeValues[i], uvPolygons)) {
      #pragma omp critical
      {
        overlappingFaces.insert(&meshData->faces[i]);
      }
    }
  }

  return overlappingFaces;
}

vector<unordered_set<BFacePtr>>
getFaceClusters(const unordered_set<BFacePtr>& bmFaces) {
  vector<unordered_set<BFacePtr>> clusters;
  unordered_set<BFacePtr> faceSet(bmFaces.begin(), bmFaces.end());

  while (!faceSet.empty()) {
    unordered_set<BFacePtr> cluster;

    BFacePtr bmFace = faceSet.extract(faceSet.begin()).value();
    set<BFacePtr> queue;
    queue.insert(bmFace);

    while (!queue.empty()) {
      bmFace = *prev(queue.end());
      queue.erase(prev(queue.end()));

      cluster.insert(bmFace);
      faceSet.erase(bmFace);

      auto adjacent = getAdjacentFaces(bmFace);
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

vector<size_t> getOverlappingClusters(py::object& bm,
                                      const polyList& uvPolygons,
                                      py::object& bUVIslands) {
  auto timeMeshData = timer("BMesh data transfer:");

  auto meshData = getMeshData(bm);

  timeMeshData();

  auto timeOverlap = timer("Overlap detection:");

  vector<boxValue> rtreeValues;

  size_t i = 0;
  transform(uvPolygons.begin(),
            uvPolygons.end(),
            back_inserter(rtreeValues),
            [&](const poly& p) {
              return make_pair(bg::return_envelope<box>(p), i++);
            });

  bgi::rtree<boxValue, bgi::rstar<8>> rtree(rtreeValues);

  auto overlappingFaces =
      getOverlappingFaces(meshData, uvPolygons, rtreeValues, rtree);
  auto faceClusters = getFaceClusters(overlappingFaces);

  vector<unordered_set<BFacePtr>> uvIslands;
  for (const auto& i : bUVIslands) {
    unordered_set<BFacePtr> island;
    for (const auto& fi : i) {
      island.insert(&meshData->faces.at(fi.cast<size_t>()));
    }

    uvIslands.push_back(island);
  }
  vector<set<BEdgePtr>> uvIslandBorders;
  for (const auto& i : uvIslands) {
    auto be = getBorderEdges(i);
    set<BEdgePtr> border = set<BEdgePtr>(be.begin(), be.end());
    uvIslandBorders.push_back(border);
  }
  vector<set<BEdgePtr>> faceClusterBorders;
  for (const auto& fc : faceClusters) {
    auto b = getBorderEdges(fc);
    auto border = set<BEdgePtr>(b.begin(), b.end());
    faceClusterBorders.push_back(border);
  }
  map<size_t, size_t> faceClusterToIsland;
  for (size_t i = 0; i < faceClusters.size(); ++i) {
    auto it = find_if(uvIslands.begin(), uvIslands.end(), [&](const auto& island) {
      return island.find(*faceClusters[i].begin()) != island.end();
    });

    if(it != uvIslands.end()) {
      faceClusterToIsland[i] = distance(uvIslands.begin(), it);
    }
  }
  vector<double> faceClusterIslandBoundaryLengths;
  for (size_t i = 0; i < faceClusters.size(); ++i) {
    auto& fcBorder = faceClusterBorders[i];
    auto& uvIslandBorder = uvIslandBorders[faceClusterToIsland[i]];

    set<BEdgePtr> connectedBorder;
    set_difference(fcBorder.begin(),
                   fcBorder.end(),
                   uvIslandBorder.begin(),
                   uvIslandBorder.end(),
                   inserter(connectedBorder, connectedBorder.begin()));

    double length = 0;
    for (const auto& e : connectedBorder) {
      length += e->length;
    }

    faceClusterIslandBoundaryLengths.push_back(length);
  }
  vector<size_t> sortedFaceClusters(faceClusters.size());
  iota(sortedFaceClusters.begin(), sortedFaceClusters.end(), 0);
  sort(sortedFaceClusters.begin(),
       sortedFaceClusters.end(),
       [&](const auto& a, const auto& b) {
         return faceClusterIslandBoundaryLengths[a] > faceClusterIslandBoundaryLengths[b];
       });

  unordered_set<size_t> activePolys;
  vector<size_t> clustersToRemove;
  for (const auto& c : sortedFaceClusters) {
    if (isPolygonIntersecting(rtree,
                              rtreeValues[(*faceClusters[c].begin())->index],
                              uvPolygons,
                              true,
                              activePolys)) {
      clustersToRemove.push_back(c);

      continue;
    }

    for (const auto& f : faceClusters[c]) {
      activePolys.insert(rtreeValues[f->index].second);
    }
  }

  vector<size_t> outFaces;
  for (const auto& ci : clustersToRemove) {
    for (const auto& f : faceClusters[ci]) {
      outFaces.push_back(f->index);
    }
  }

  timeOverlap();

  return outFaces;
}

void pybind11OverlapRemove(py::module& m) {
  py::class_<poly>(m, "poly")
      .def(py::init<>())
      .def("addPoint", [](poly& p, const double& x, const double& y) {
        p.outer().push_back(point(x, y));
      });

  py::bind_vector<polyList>(m, "polyList");

  m.def("getOverlappingClusters", getOverlappingClusters);
}
