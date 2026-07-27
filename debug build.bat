@echo off
setlocal

set buildDir=build\windows

cmake --build %buildDir% --config Debug
cmake --install %buildDir% --config Debug
