// Copyright (C) 2023-2026 Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "structures.h"

#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

#include <vector>
#include <set>
#include <unordered_map>
#include <string>
#include <type_traits>
#include <functional>

#include <boost/container_hash/hash.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace std;

inline
bool isZero(const double& x,
            const double& epsilon = std::numeric_limits<double>::epsilon()) {
    return abs(x) < epsilon;
}

inline BVertPtr getEdgeOtherVert(const BEdgePtr& edge,
                                 const BVertPtr& vert) {
  for (auto& v : edge->verts) {
    if(v != vert) {
      return v;
    }
  }

  return nullptr;
}

template <typename T>
vector<vector<T>> getCombinations(const vector<T>& v,
                                  const size_t& r) {
  vector<vector<T>> out;

  size_t n = v.size();
  if(r > n) {
    return {};
  }

  vector<size_t> indices;
  for(size_t i = 0; i < r; i++) {
    indices.push_back(i);
  }

  vector<T> tmp;
  for(const auto& i : indices) {
    tmp.push_back(v[i]);
  }
  out.push_back(tmp);

  for (int i = r - 1; i >= 0; i--) {
    if (indices[i] != i + n - r) {
      indices[i] += 1;
      for (int j = i + 1; j < r; j++) {
        indices[j] = indices[j - 1] + 1;
      }

      vector<T> tmp2;
      for (const auto &i : indices) {
        tmp2.push_back(v[i]);
      }
      out.push_back(tmp2);

      i = r;
    }
  }

  return out;
}

template <typename T>
set<T> getIntersection(const set<T>& a, const set<T>& b) {
  set<T> result;
  set_intersection(a.begin(), a.end(),
                   b.begin(), b.end(),
                   inserter(result, result.end()));

  return result;
}

template <typename T>
vector<T> getIntersection(vector<T> a, vector<T> b) {
  sort(a.begin(), a.end());
  sort(b.begin(), b.end());

  vector<T> result;
  set_intersection(a.begin(), a.end(),
                   b.begin(), b.end(),
                   back_inserter(result));

  return result;
}

template <typename T, typename F>
xt::xarray<double> getVector2Xarray(const T& v, size_t size, F getFn) {
  vector<double> tmp;
  for(const auto& n : v) {
    auto g = getFn(n);
    for(size_t i = 0; i < size; i++) {
      tmp.push_back(g[i]);
    }
  }

  xt::xarray<double> out = xt::adapt(tmp);
  out.reshape({v.size(), size});

  return out;
}

inline
vector3df
getVector3df(const pybind11::object& obj,
             const char* str) {
  return xt::adapt(obj.attr(str).cast<array<double, 3>>());
}

template <typename Fn,
          typename FnArg,
          typename... FnArgs>
struct CachedFunction {
  using ReturnType = invoke_result_t<Fn, FnArg, FnArgs...>;
  using CacheType =
      conditional_t<(sizeof...(FnArgs) > 0),
                    unordered_map<tuple<FnArg, FnArgs...>,
                                  ReturnType,
                                  boost::hash<tuple<FnArg, FnArgs...>>>,
                    unordered_map<FnArg, ReturnType>>;

  Fn fn;
  CacheType cache;

  template <typename Arg, typename... Args>
  ReturnType operator()(const Arg& arg, const Args&... args) {
    auto obj = make_tuple(arg, args...);

    if (cache.find(obj) == cache.end()) {
      cache[obj] = fn(arg, args...);
    }

    return cache[obj];
  }

  template <typename Arg>
  ReturnType operator()(const Arg& arg) {
    if (cache.find(arg) == cache.end()) {
      cache[arg] = fn(arg);
    }

    return cache[arg];
  }
};

template <typename Fn>
struct CachedFunction<Fn, pybind11::object> {
  using ReturnType = invoke_result_t<Fn, pybind11::object>;

  Fn fn;
  unordered_map<pybind11::object, ReturnType, PyObjectHash, PyObjectEqual> cache;

  ReturnType operator()(const pybind11::object& arg) {
    if (cache.find(arg) == cache.end()) {
      cache[arg] = fn(arg);
    }

    return cache[arg];
  }
};

template <typename Fn,
          typename... FnArgs>
struct CachedFunctionOrderless {
  using ReturnType = invoke_result_t<Fn, FnArgs...>;

  Fn fn;
  unordered_map<pair<FnArgs...>,
                ReturnType,
                boost::hash<pair<FnArgs...>>> cache;

  template <typename A, typename B>
  ReturnType operator()(const A& a, const B& b) {
    pair<A, B> obj;
    if (a < b) {
      obj = make_pair(a, b);
    } else {
      obj = make_pair(b, a);
    }

    if (cache.find(obj) == cache.end()) {
      cache[obj] = fn(a, b);
    }

    return cache[obj];
  }
};
