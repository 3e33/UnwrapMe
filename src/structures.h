// Copyright (C) 2023-2026 Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <map>
#include <memory>
#include <queue>
#include <set>
#include <vector>

#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xfixed.hpp>

#include <pybind11/stl_bind.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;

const double RAD = xt::numeric_constants<double>::PI / 180;
const double DEG = 180 / xt::numeric_constants<double>::PI;

struct Bmesh;
struct BVert;
struct BEdge;
struct BFace;
using BmeshPtr = shared_ptr<Bmesh>;
using BVertPtr = BVert*;
using BEdgePtr = BEdge*;
using BFacePtr = BFace*;

struct Chart;
struct Face;
using ChartPtr = shared_ptr<Chart>;
using FacePtr = shared_ptr<Face>;

struct UnfitFace;
struct Hole;
using UnfitFacePtr = shared_ptr<UnfitFace>;
using HolePtr = shared_ptr<Hole>;
using AllocMap = vector<int>;
using UnfitMap = map<size_t, UnfitFacePtr>;

using matrix4d = xt::xtensor_fixed<double, xt::xshape<4, 4>>;
using matrix6df = xt::xtensor_fixed<double, xt::xshape<6, 6>>;
using matrix5df = xt::xtensor_fixed<double, xt::xshape<5, 5>>;
using matrix4df = xt::xtensor_fixed<double, xt::xshape<4, 4>>;
using matrix3df = xt::xtensor_fixed<double, xt::xshape<3, 3>>;
using matrix2df = xt::xtensor_fixed<double, xt::xshape<2, 2>>;
using vector4d = xt::xtensor<double, 1>;
using vector3d = xt::xtensor<double, 1>;
using vector2d = xt::xtensor<double, 1>;
using vector5df = xt::xtensor_fixed<double, xt::xshape<5>>;
using vector4df = xt::xtensor_fixed<double, xt::xshape<4>>;
using vector3df = xt::xtensor_fixed<double, xt::xshape<3>>;
using vector2df = xt::xtensor_fixed<double, xt::xshape<2>>;

struct BVert {
  size_t index;
  vector3d co;
  vector<BEdgePtr> link_edges;
  bool select;
  vector3d normal;

  bool operator<(BVertPtr &other) { return index < other->index; }
};

struct BEdge {
  size_t index;
  double length;
  double faceAngle;
  bool seam;
  vector<BVertPtr> verts;
  vector<BFacePtr> link_faces;
  bool select;

  bool operator<(BEdgePtr &other) { return index < other->index; }
};

struct BFace {
  size_t index;
  double area;
  vector3d centroid;
  vector<BEdgePtr> edges;
  vector<BVertPtr> verts;
  bool select;
  vector3d normal;
  bool hide;

  bool operator<(BFacePtr &other) { return index < other->index; }
};

struct Bmesh {
  vector<BFace> faces;
  vector<BEdge> edges;
  vector<BVert> verts;
};

struct HashFns {
  size_t operator()(const BFacePtr &bf) const noexcept { return bf->index; }

  size_t operator()(const tuple<BFacePtr, BFacePtr> &bft) const noexcept {
    auto i1 = get<0>(bft)->index;
    auto i2 = get<1>(bft)->index;
    if (i1 < i2) {
      auto h1 = hash<unsigned int>{}(i1);
      auto h2 = hash<unsigned int>{}(i2);

      return h1 ^ (h2 << 1);
    } else {
      auto h1 = hash<unsigned int>{}(i2);
      auto h2 = hash<unsigned int>{}(i1);

      return h1 ^ (h2 << 1);
    }
  }
};

// A*

struct Node {
  BVertPtr bmVert = nullptr;
  BEdgePtr bmEdge = nullptr;
  double cost = 0.0;
  shared_ptr<Node> parentNode = nullptr;
  double travelledDistance = 0.0;

  bool operator()(const shared_ptr<Node> &a, const shared_ptr<Node> &b) {
    return a->cost > b->cost;
  }
};

// Charts

struct Face {
  BFacePtr bmFace;
  double cost = 0;
  double seedDistance;
  double fittingError;

  bool operator()(const shared_ptr<Face> &a, const shared_ptr<Face> &b) {
    return a->cost > b->cost;
  }
};

struct Chart {
  size_t index;
  priority_queue<FacePtr, vector<FacePtr>, Face> queue;
  vector4d proxy;
  double area;
  vector<BFacePtr> bmFaces;
  set<BEdgePtr> borderEdges;
};

struct UnfitFace {
  size_t chartIndex;
  shared_ptr<Face> face;
};

struct Hole {
  vector<BFacePtr> bmFaces;
  double area;
};

// PyBind
struct PyObjectHash {
  size_t operator()(const pybind11::object& obj) const { return pybind11::hash(obj); }
};

struct PyObjectEqual {
  bool operator()(const pybind11::object& obj1, const pybind11::object& obj2) const {
    return obj1.is(obj2);
  }
};

using PyObjectSet = unordered_set<pybind11::object, PyObjectHash, PyObjectEqual>;

template <typename T>
using PyObjectMap =
  unordered_map<pybind11::object, T, PyObjectHash, PyObjectEqual>;

