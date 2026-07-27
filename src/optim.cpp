// Copyright (C) 2024- Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#include "optim.h"

#include "dev.h"
#include "structures.h"
#include "common.h"
#include "utilities.h"

#include <xtensor/xbuilder.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xrandom.hpp>

#include <omp.h>

#include <iomanip>
#include <random>
#include <cmath>

namespace Optim {
  pair<vector4df, matrix4df>
  getDerivatives(const vector4df& x,
                 const double& totalArea,
                 const vector<BFacePtr>& bmFaces) {
    vector4df g = xt::zeros<double>({4});
    matrix4df h = xt::zeros<double>({4,4});
    double cosx3 = std::cos(x[3]);
    double sinx3 = std::sin(x[3]);

#pragma omp parallel
    {
      vector4df g_ = xt::zeros<double>({4});
      matrix4df h_ = xt::zeros<double>({4, 4});

#pragma omp for
      for (int i = 0; i < bmFaces.size(); ++i) {
        auto& n = bmFaces[i]->normal;
        double a = bmFaces[i]->area / totalArea;

        g_[0] += 2 * n[0] * n[0] * x[0] + 2 * n[0] * n[1] * x[1] +
          2 * n[0] * n[2] * x[2] - 2 * n[0] * cosx3 +
          4 * x[0] * x[0] * x[0] + 4 * x[0] * x[1] * x[1] +
          4 * x[0] * x[2] * x[2] - 4 * x[0];
        g_[1] += 2 * n[0] * n[1] * x[0] + 2 * n[1] * n[1] * x[1] +
          2 * n[1] * n[2] * x[2] - 2 * n[1] * cosx3 +
          4 * x[0] * x[0] * x[1] + 4 * x[1] * x[1] * x[1] +
          4 * x[1] * x[2] * x[2] - 4 * x[1];
        g_[2] += 2 * n[0] * n[2] * x[0] + 2 * n[1] * n[2] * x[1] +
          2 * n[2] * n[2] * x[2] - 2 * n[2] * cosx3 +
          4 * x[0] * x[0] * x[2] + 4 * x[1] * x[1] * x[2] +
          4 * x[2] * x[2] * x[2] - 4 * x[2];
        g_[3] += 2 * n[0] * x[0] * sinx3 + 2 * n[1] * x[1] * sinx3 +
          2 * n[2] * x[2] * sinx3 - 2 * sinx3 * cosx3;

        g_ *= a;

        h_(0, 0) += 2 * n[0] * n[0] + 12 * x[0] * x[0] + 4 * x[1] * x[1] +
          4 * x[2] * x[2] - 4;
        h_(0, 1) += 2 * n[0] * n[1] + 8 * x[0] * x[1];
        h_(0, 2) += 2 * n[0] * n[2] + 8 * x[0] * x[2];
        h_(0, 3) += 2 * n[0] * sinx3;
        h_(1, 0) += 2 * n[0] * n[1] + 8 * x[0] * x[1];
        h_(1, 1) += 2 * n[1] * n[1] + 4 * x[0] * x[0] + 12 * x[1] * x[1] +
          4 * x[2] * x[2] - 4;
        h_(1, 2) += 2 * n[1] * n[2] + 8 * x[1] * x[2];
        h_(1, 3) += 2 * n[1] * sinx3;
        h_(2, 0) += 2 * n[0] * n[2] + 8 * x[0] * x[2];
        h_(2, 1) += 2 * n[1] * n[2] + 8 * x[1] * x[2];
        h_(2, 2) += 2 * n[2] * n[2] + 4 * x[0] * x[0] + 4 * x[1] * x[1] +
          12 * x[2] * x[2] - 4;
        h_(2, 3) += 2 * n[2] * sinx3;
        h_(3, 0) += 2 * n[0] * sinx3;
        h_(3, 1) += 2 * n[1] * sinx3;
        h_(3, 2) += 2 * n[2] * sinx3;
        h_(3, 3) += 2 * n[0] * x[0] * cosx3 + 2 * n[1] * x[1] * cosx3 +
          2 * n[2] * x[2] * cosx3 + 2 * sinx3 * sinx3 -
          2 * cosx3 * cosx3;

        h_ *= a;
      }

      #pragma omp critical
      {
        g += g_;
        h += h_;
      }
    }

    return make_pair(g, h);
  }

  template <typename T>
  vector4df getOptimProxy(vector4df x,
                          const T& bmFaces,
                          const double& totalArea,
                          const int& k,
                          double globalTemp) {
    matrix4df I = xt::eye(4);

    vector4df x_ = x;
    auto [g_, H_] = getDerivatives(x_, totalArea, bmFaces);
    double r = 0.0001;
    vector4df d = xt::linalg::solve(H_ + (r * I), g_);

    x = x - d;

    double ng = xt::norm_l2(g_)();

    double sigma = 1.0;
    double maxRand = 0.01;

    for (int i = 0; i < k; ++i) {
      if (xt::norm_l2(x - x_)() < 0.00001) {
        break;
      }

      auto [g, H] = getDerivatives(x, totalArea, bmFaces);

      vector4df dx = x - x_;
      double vdx = xt::norm_sq(dx)();
      double r_n =
          isZero(vdx) ? 0.0
                      : xt::norm_l2(g - g_ - (xt::linalg::dot(H_, dx)))() / vdx;
      r = max(r_n, r * 0.5);

      ng = xt::norm_l2(g)();

      matrix4df ri = std::sqrt(r * ng) * I;
      d = xt::linalg::solve(H + ri, g);

      sigma = sigma * std::pow(0.5, i) * globalTemp;
      vector4df noise = xt::random::randn<double>({4}, 0.0, sigma);
      noise = xt::clip(noise, -maxRand, maxRand);

      d = d + noise;

      x_ = x;
      g_ = g;
      H_ = H;

      x = x - d;
    }

    return x;
  }

  template <typename T>
  vector4df getProxy(const vector4df& proxyGuess,
                     const T& bmFaces,
                     double globalTemp) {
    double totalArea = getFacesArea(bmFaces);

    vector4df proxy = getOptimProxy(proxyGuess, bmFaces, totalArea, 300, globalTemp);
    proxy[3] = fmod(proxy[3], xt::numeric_constants<double>::PI);

    double l = xt::norm_l2(xt::view(proxy, xt::range(0, 3)))();
    if (!isZero(l)) {
      xt::view(proxy, xt::range(0, 3)) /= l;
    }

    return proxy;
  }

  template <>
  vector4df getProxy(const vector4df& proxyGuess,
                     const unordered_set<BFacePtr>& bmFaces,
                     double globalTemp) {
    double totalArea = getFacesArea(bmFaces);

    vector<BFacePtr> faces(bmFaces.begin(), bmFaces.end());

    vector4df proxy = getOptimProxy(proxyGuess, faces, totalArea, 300, globalTemp);
    proxy[3] = fmod(proxy[3], xt::numeric_constants<double>::PI);

    double l = xt::norm_l2(xt::view(proxy, xt::range(0, 3)))();
    if (!isZero(l)) {
      xt::view(proxy, xt::range(0, 3)) /= l;
    }

    return proxy;
  }

  template vector4df
  getProxy<vector<BFacePtr>>(const vector4df& proxyGuess,
                             const vector<BFacePtr>& bmFaces,
                             double globalTemp);
  pair<vector3df, matrix3df>
  getDerivatives(const vector3df& x,
                 const double& angle,
                 const double& totalArea,
                 const vector<BFacePtr>& bmFaces) {
    vector3df g = xt::zeros<double>({3});
    matrix3df h = xt::zeros<double>({3,3});
    double cosx3 = std::cos(angle);
    double sinx3 = std::sin(angle);

#pragma omp parallel
    {
      vector3df g_ = xt::zeros<double>({3});
      matrix3df h_ = xt::zeros<double>({3, 3});

#pragma omp for
      for (int i = 0; i < bmFaces.size(); ++i) {
        auto& n = bmFaces[i]->normal;
        double a = bmFaces[i]->area / totalArea;

        g_[0] += 2 * n[0] * n[0] * x[0] + 2 * n[0] * n[1] * x[1] +
                 2 * n[0] * n[2] * x[2] - 2 * n[0] * cosx3 +
                 4 * x[0] * x[0] * x[0] + 4 * x[0] * x[1] * x[1] +
                 4 * x[0] * x[2] * x[2] - 4 * x[0];
        g_[1] += 2 * n[0] * n[1] * x[0] + 2 * n[1] * n[1] * x[1] +
                 2 * n[1] * n[2] * x[2] - 2 * n[1] * cosx3 +
                 4 * x[0] * x[0] * x[1] + 4 * x[1] * x[1] * x[1] +
                 4 * x[1] * x[2] * x[2] - 4 * x[1];
        g_[2] += 2 * n[0] * n[2] * x[0] + 2 * n[1] * n[2] * x[1] +
                 2 * n[2] * n[2] * x[2] - 2 * n[2] * cosx3 +
                 4 * x[0] * x[0] * x[2] + 4 * x[1] * x[1] * x[2] +
                 4 * x[2] * x[2] * x[2] - 4 * x[2];

        g_ *= a;

        h_(0, 0) += 2 * n[0] * n[0] + 12 * x[0] * x[0] + 4 * x[1] * x[1] +
                    4 * x[2] * x[2] - 4;
        h_(0, 1) += 2 * n[0] * n[1] + 8 * x[0] * x[1];
        h_(0, 2) += 2 * n[0] * n[2] + 8 * x[0] * x[2];
        h_(1, 0) += 2 * n[0] * n[1] + 8 * x[0] * x[1];
        h_(1, 1) += 2 * n[1] * n[1] + 4 * x[0] * x[0] + 12 * x[1] * x[1] +
                    4 * x[2] * x[2] - 4;
        h_(1, 2) += 2 * n[1] * n[2] + 8 * x[1] * x[2];
        h_(2, 0) += 2 * n[0] * n[2] + 8 * x[0] * x[2];
        h_(2, 1) += 2 * n[1] * n[2] + 8 * x[1] * x[2];
        h_(2, 2) += 2 * n[2] * n[2] + 4 * x[0] * x[0] + 4 * x[1] * x[1] +
                    12 * x[2] * x[2] - 4;

        h_ *= a;
      }

      #pragma omp critical
      {
        g += g_;
        h += h_;
      }
    }

    return make_pair(g, h);
  }

  template <typename T>
  vector3df getOptimProxy(vector3df x,
                          const double& angle,
                          const T& bmFaces,
                          const double& totalArea,
                          const int& k) {
    matrix3df I = xt::eye(3);

    vector3df x_ = x;
    auto [g_, H_] = getDerivatives(x_, angle, totalArea, bmFaces);
    double r = 0.0001;
    vector3df d = xt::linalg::solve(H_ + (r * I), g_);

    x = x - d;

    double ng = xt::norm_l2(g_)();

    for (int i = 0; i < k; ++i) {
      if (xt::norm_l2(x - x_)() < 0.00001) {
        break;
      }

      auto [g, H] = getDerivatives(x, angle, totalArea, bmFaces);

      vector3df dx = x - x_;
      double vdx = xt::norm_sq(dx)();
      double r_n =
          isZero(vdx) ? 0.0
                      : xt::norm_l2(g - g_ - (xt::linalg::dot(H_, dx)))() / vdx;
      r = max(r_n, r * 0.5);

      ng = xt::norm_l2(g)();

      matrix3df ri = std::sqrt(r * ng) * I;
      d = xt::linalg::solve(H + ri, g);

      x_ = x;
      g_ = g;
      H_ = H;

      x = x - d;
    }

    return x;
  }

  template <typename T>
  vector4df getProxy(const vector3df& proxyGuess,
                     const double& angle,
                     const T& bmFaces) {
    double totalArea = getFacesArea(bmFaces);

    vector3df proxy = getOptimProxy(proxyGuess, angle, bmFaces, totalArea, 300);

    double l = xt::norm_l2(proxy)();
    if (!isZero(l)) {
      proxy /= l;
    }

    return vector4df{proxy[0], proxy[1], proxy[2], angle};
  }

  template <>
  vector4df getProxy(const vector3df& proxyGuess,
                     const double& angle,
                     const unordered_set<BFacePtr>& bmFaces) {
    double totalArea = getFacesArea(bmFaces);

    vector<BFacePtr> faces(bmFaces.begin(), bmFaces.end());

    vector3df proxy = getOptimProxy(proxyGuess, angle, faces, totalArea, 300);

    double l = xt::norm_l2(proxy)();
    if (!isZero(l)) {
      proxy /= l;
    }

    return vector4df{proxy[0], proxy[1], proxy[2], angle};
  }

  template vector4df
  getProxy<vector<BFacePtr>>(const vector3df& proxyGuess,
                             const double& angle,
                             const vector<BFacePtr>& bmFaces);

} // namespace Optim
