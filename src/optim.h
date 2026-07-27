// Copyright (C) 2024- Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "structures.h"

#include <xtensor/xarray.hpp>
#include <xtensor/xshape.hpp>

namespace Optim {
  template <typename T>
  vector4df getProxy(const vector4df& proxyGuess,
                     const T& bmFaces,
                     double globalTemp = 1);

  template <typename T>
  vector4df getProxy(const vector3df& proxyGuess,
                     const double& angle,
                     const T& bmFaces);
} // namespace Optim
