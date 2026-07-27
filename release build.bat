@echo off
setlocal

set buildDir=build\windows

cmake --build %buildDir% --config Release
cmake --install %buildDir% --config Release
