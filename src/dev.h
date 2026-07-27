// Copyright (C) 2023-2026 Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <iostream>
#include <vector>
#include <type_traits>

inline auto timer(const std::string s = "") {
  auto start = std::chrono::high_resolution_clock::now();

  return [=]() {
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(stop - start);

    std::cout << s << " ";
    std::cout << std::setprecision(5) << duration.count() << "s" << std::endl;
  };
}

namespace Detail {
  template<typename, typename = void>
  constexpr bool is_iterable = false;

  template<typename T>
  constexpr bool is_iterable<
      T,
      std::void_t<decltype(std::declval<T>().begin()),
                  decltype(std::declval<T>().end())
      >
  > = true;

  template <typename T, typename Stream>
  std::enable_if_t<!is_iterable<T>>
  print(const T& o, Stream& stream) {
    stream << o << std::endl;
  }

  template <typename T, typename Stream>
  std::enable_if_t<is_iterable<T>>
  print(const T& o, Stream& stream) {
    stream << "{";
    if (!o.empty()) {
      auto it = o.cbegin();
      stream << *it;

      while (o.cend() != ++it) {
        stream << ", " << *it;
      }
    }
    stream << "}" << std::endl;
  }
}

template <typename T>
void oprint(const T& o) {
  Detail::print(o, std::cout);
}

template <typename T>
void eprint(const T& o) {
  Detail::print(o, std::cerr);
}
