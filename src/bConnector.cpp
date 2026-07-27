// Copyright (C) 2023-2026 Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#include "bConnector.h"

#include "structures.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>

#include <memory>
#include <algorithm>
#include <iterator>
#include <cmath>

using namespace std;

namespace py = pybind11;

BmeshPtr getFastcoreBmesh(const py::object& bmesh) {
  auto fcBmesh = make_shared<Bmesh>();
  fcBmesh->faces.resize(py::len(bmesh.attr("faces")));
  fcBmesh->edges.resize(py::len(bmesh.attr("edges")));
  fcBmesh->verts.resize(py::len(bmesh.attr("verts")));

  generate(fcBmesh->faces.begin(), fcBmesh->faces.end(), []() {
    return BFace();
  });

  generate(fcBmesh->edges.begin(), fcBmesh->edges.end(), []() {
    return BEdge();
  });

  generate(fcBmesh->verts.begin(), fcBmesh->verts.end(), []() {
    return BVert();
  });

  for (const auto& face : bmesh.attr("faces")) {
    size_t index = face.attr("index").cast<size_t>();
    auto& f = fcBmesh->faces[index];

    f.index = index;
    f.area = face.attr("calc_area")().cast<double>();
    f.centroid = xt::adapt(
        face.attr("calc_center_median_weighted")().cast<array<double, 3>>());
    f.select = face.attr("select").cast<bool>();
    f.normal = xt::adapt(face.attr("normal").cast<array<double, 3>>());
    f.hide = face.attr("hide").cast<bool>();

    for (const auto& e : face.attr("edges")) {
      auto ei = e.attr("index").cast<size_t>();
      f.edges.push_back(&fcBmesh->edges[ei]);
    }

    for (const auto& v : face.attr("verts")) {
      auto vi = v.attr("index").cast<size_t>();
      f.verts.push_back(&fcBmesh->verts[vi]);
    }
  }

  for (const auto& edge : bmesh.attr("edges")) {
    size_t index = edge.attr("index").cast<size_t>();
    auto& e = fcBmesh->edges[index];

    e.index = index;
    e.length = edge.attr("calc_length")().cast<double>();
    e.seam = edge.attr("seam").cast<bool>();
    e.select = edge.attr("select").cast<bool>();
    e.faceAngle = edge.attr("calc_face_angle")(INFINITY).cast<double>();

    for (const auto& v : edge.attr("verts")) {
      auto vi = v.attr("index").cast<size_t>();
      e.verts.push_back(&fcBmesh->verts[vi]);
    }

    for (const auto& f : edge.attr("link_faces")) {
      auto fi = f.attr("index").cast<size_t>();
      e.link_faces.push_back(&fcBmesh->faces[fi]);
    }
  }

  for (const auto& vert : bmesh.attr("verts")) {
    size_t index = vert.attr("index").cast<size_t>();
    auto& v = fcBmesh->verts[index];

    v.index = index;
    v.co = xt::adapt(vert.attr("co").cast<array<double, 3>>());
    v.select = vert.attr("select").cast<bool>();
    v.normal = xt::adapt(vert.attr("normal").cast<array<double, 3>>());

    for (const auto& e : vert.attr("link_edges")) {
      auto ei = e.attr("index").cast<size_t>();
      v.link_edges.push_back(&fcBmesh->edges[ei]);
    }
  }

  return fcBmesh;
}

void updateBmesh(py::object& bm, BmeshPtr fcBmesh) {
  for (auto& e : fcBmesh->edges) {
    // if(e.seam) {
    //   bm.attr("edges").attr("__getitem__")(e.index).attr("seam") = true;
    // }

    // if(e.select) {
    //   bm.attr("edges").attr("__getitem__")(e.index).attr("select") = true;
    // }

    bm.attr("edges").attr("__getitem__")(e.index).attr("seam") = e.seam;
  }

  // for (auto& f : fcBmesh->faces) {
  //   if(f->select) {
  //     bm.attr("faces").attr("__getitem__")(f->index).attr("select") = true;
  //   }
  // }

  // for (auto& v : fcBmesh->verts) {
  //   if(v->select) {
  //     bm.attr("verts").attr("__getitem__")(v->index).attr("select") = true;
  //   }
  // }
}
