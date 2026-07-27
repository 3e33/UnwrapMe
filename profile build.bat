@echo off
setlocal

set buildDir=build\windows

cmake --build %buildDir% --config RelWithDebInfo
cmake --install %buildDir% --config RelWithDebInfo 
