// Copyright (C) 2023-2026 Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "structures.h"
#include "utilities.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <chrono>
#include <unordered_set>
#include <vector>

#include <xtensor/xexpression.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xnorm.hpp>

#include "xtl/xcomplex.hpp"
#include "xtensor/xcomplex.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include <xtensor/xsort.hpp>
#include <xtensor/xadapt.hpp>

#include "xtensor-blas/xblas.hpp"
#include "xtensor-blas/xlapack.hpp"
#include "xtensor-blas/xblas_utils.hpp"
#include <xtensor-blas/xlinalg.hpp>

namespace Fitting {
  template <typename T>
  using MNPair = pair<T, T>;

  vector3df
  getSurfaceRotation(const unordered_set<BVertPtr>& bmVerts,
                     const vector3df& centre);

  template <class E, class W, class V>
  int ggev(E& A,
           E& B,
           char jobvl,
           char jobvr,
           W& alphar,
           W& alphai,
           W& beta,
           V& VL,
           V& VR) {
    XTENSOR_ASSERT(A.dimension() == 2);
    XTENSOR_ASSERT(A.layout() == layout_type::column_major);
  
    using value_type = typename E::value_type;
  
    const auto N = A.shape()[0];
    xt::uvector<value_type> work(1);
  
    int info =
        cxxlapack::ggev<xt::blas_index_t>(jobvl,
                                          jobvr,
                                          static_cast<xt::blas_index_t>(N),
                                          A.data(),
                                          stride_back(A),
                                          B.data(),
                                          stride_back(B),
                                          alphar.data(),
                                          alphai.data(),
                                          beta.data(),
                                          VL.data(),
                                          stride_back(VL),
                                          VR.data(),
                                          stride_back(VR),
                                          work.data(),
                                          static_cast<xt::blas_index_t>(-1));
  
    if (info != 0) {
      XTENSOR_THROW(std::runtime_error,
                    "Could not find workspace size for geev.");
    }
  
    work.resize(static_cast<std::size_t>(work[0]));
  
    info = cxxlapack::ggev<xt::blas_index_t>(
        jobvl,
        jobvr,
        static_cast<xt::blas_index_t>(N),
        A.data(),
        stride_back(A),
        B.data(),
        stride_back(B),
        alphar.data(),
        alphai.data(),
        beta.data(),
        VL.data(),
        stride_back(VL),
        VR.data(),
        stride_back(VR),
        work.data(),
        static_cast<xt::blas_index_t>(work.size()));
  
    return info;
  }

  template <class E>
  auto getGeneralisedEigenvectors(const xt::xexpression<E>& A,
                                  const xt::xexpression<E>& B) {
    using value_type = typename E::value_type;
  
    assert_nd_square(A);
    assert_nd_square(B);
    auto M = xt::copy_to_layout<xt::layout_type::column_major>(A.derived_cast());
    auto M2 = xt::copy_to_layout<xt::layout_type::column_major>(B.derived_cast());
  
    std::size_t N = M.shape()[0];
    std::array<std::size_t, 1> vN = {N};
    xt::xtensor<value_type, 1, xt::layout_type::column_major> alphar(vN),
        alphai(vN), beta(vN);
  
    std::array<std::size_t, 2> shp = {N, N};
    xt::xtensor<value_type, 2, xt::layout_type::column_major> VL(shp), VR(shp);
  
    int info = ggev(M, M2, 'N', 'V', alphar, alphai, beta, VL, VR);
  
    if (info != 0) {
      XTENSOR_THROW(std::runtime_error,
                    "Eigenvalue calculation did not converge.");
    }
  
    // normalise
    for (size_t i = 0; i < VR.shape()[0]; ++i) {
      xt::col(VR, i) /= xt::norm_l2(xt::col(VR, i));
    }
  
    return VR;
  }
  template <class E, class E2>
  auto getTaubinError(const xt::xexpression<E>& coefficients_,
                      const xt::xexpression<E2>& M_,
                      const xt::xexpression<E2>& N_) {
    auto coefficients = coefficients_.derived_cast();
    auto M = M_.derived_cast();
    auto N = N_.derived_cast();

    auto z1 = xt::linalg::vdot(coefficients, xt::linalg::dot(M, coefficients));
    auto z2 = xt::linalg::vdot(coefficients, xt::linalg::dot(N, coefficients));

    if (!isZero(z2)) {
      return abs(z1 / z2);
    }

    return std::numeric_limits<double>::infinity();
  }

  template <class E, class E2>
  auto getArgSortedTaubinErrors(const xt::xexpression<E>& eigenVectors_,
                                const xt::xexpression<E2>& M_,
                                const xt::xexpression<E2>& N_) {
    auto eigenVectors = eigenVectors_.derived_cast();
    auto M = M_.derived_cast();
    auto N = N_.derived_cast();

    vector<double> errors;
    std::size_t n = eigenVectors.shape()[0];
    errors.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      auto v = xt::row(eigenVectors, i);
      errors.push_back(getTaubinError(v, M, N));
    }

    return xt::argsort(xt::adapt(errors));
  }
} // namespace Fitting
