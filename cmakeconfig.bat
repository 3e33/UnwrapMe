@echo off
setlocal

set buildDir=build\windows

cmake -S . -B %buildDir% -G "Ninja Multi-Config" -DCMAKE_C_COMPILER="cl" -DCMAKE_CXX_COMPILER="cl" %*
