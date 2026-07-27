# Copyright (C) 2023-2026 Elias V.
# SPDX-License-Identifier: GPL-3.0-only

function(getPackage name url)
  set(destDir "${CMAKE_CURRENT_BINARY_DIR}/packages")

  if(EXISTS "${destDir}/${name}")
    message(STATUS "Skipping ${name} - package folder already exists")
    return()
  endif()

  set(tmpFile "${CMAKE_CURRENT_BINARY_DIR}/tmp/package.zip")

  message("")
  message(STATUS
    "Downloading ${name}
from ${url}")
  file(DOWNLOAD
    "${url}"
    "${tmpFile}"
    STATUS downloadStatus
  )

  list(GET downloadStatus 0 downloadCode)
  list(GET downloadStatus 1 downloadError)
  if(NOT downloadCode EQUAL 0)
    message(FATAL_ERROR "Download failed for ${name}: ${downloadError}")
  endif()

  message(STATUS "Extracting zip...")
  file(ARCHIVE_EXTRACT INPUT
    "${tmpFile}"
    DESTINATION "${destDir}"
  )

  message(STATUS "Deleting zip...")
  file(REMOVE "${tmpFile}")
endfunction()

set(packages
"boost_1_81_0\;https://archives.boost.io/release/1.81.0/source/boost_1_81_0.zip"
"OpenBLAS-0.3.21\;https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.21/OpenBLAS-0.3.21.zip"
"pybind11-2.10.3\;https://github.com/pybind/pybind11/archive/refs/tags/v2.10.3.zip"
"xsimd-10.0.0\;https://github.com/xtensor-stack/xsimd/archive/refs/tags/10.0.0.zip"
"xtensor-0.24.4\;https://github.com/xtensor-stack/xtensor/archive/refs/tags/0.24.4.zip"
"xtensor-blas-0.20.0\;https://github.com/xtensor-stack/xtensor-blas/archive/refs/tags/0.20.0.zip"
"xtensor-python-0.26.1\;https://github.com/xtensor-stack/xtensor-python/archive/refs/tags/0.26.1.zip"
"xtl-0.7.5\;https://github.com/xtensor-stack/xtl/archive/refs/tags/0.7.5.zip"
)

foreach(p IN LISTS packages)
  list(GET p 0 name)
  list(GET p 1 url)
  getPackage(
    "${name}"
    "${url}"
  )
endforeach()

message("")
message(STATUS "Done!")
