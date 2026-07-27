@echo off
setlocal

choice /c YN /m "This might take a while, continue?"
if errorlevel 2 (
exit /b
)

set projectdir=%~dp0
set depdir=%projectdir%dep\windows
set buildDir=build\windows


cd "%projectdir%packages\pybind11-2.10.3"
cmake -S . -B %buildDir% -G "Ninja"
cmake --install %buildDir% --prefix "%depdir%\pybind11"
cd "%projectdir%"


cd "%projectdir%packages\xtl-0.7.5"
cmake -S . -B %buildDir% -G "Ninja" -DCMAKE_INSTALL_PREFIX="%depdir%"
cmake --install %buildDir% --prefix "%depdir%\xtl"
cd "%projectdir%"


cd "%projectdir%packages\xtensor-0.24.4"
cmake -S . -B %buildDir% -G "Ninja" -DCMAKE_INSTALL_PREFIX="%depdir%"
cmake --install %buildDir% --prefix "%depdir%\xtensor"
cd "%projectdir%"


cd "%projectdir%packages\xsimd-10.0.0"
cmake -S . -B %buildDir% -G "Ninja" -DCMAKE_INSTALL_PREFIX="%depdir%"
cmake --install %buildDir% --prefix "%depdir%\xsimd"
cd "%projectdir%"


cd "%projectdir%packages\xtensor-python-0.26.1"
cmake -S . -B %buildDir% -G "Ninja" -DCMAKE_INSTALL_PREFIX="%depdir%"
cmake --install %buildDir% --prefix "%depdir%\xtensor-python"
cd "%projectdir%"


cd "%projectdir%packages\OpenBLAS-0.3.21"
cmake -S . -B %buildDir% -G "Ninja"^
 -DCMAKE_CXX_COMPILER=cl^
 -DCMAKE_C_COMPILER=cl^
 -DCMAKE_C_FLAGS="/W0"^
 -DDYNAMIC_ARCH=OFF^
 -DBUILD_SHARED_LIBS=OFF^
 -DCMAKE_BUILD_TYPE=Release^
 -DBUILD_TESTING=OFF^
 -DBUILD_STATIC_LIBS=ON^
 -DNOFORTRAN=ON^
 -DMSVC_STATIC_CRT=ON^
 -DCMAKE_MT=mt^
 -DTARGET=NEHALEM
cmake --build %buildDir% --config Release
cmake --install %buildDir% --prefix "%depdir%\OpenBLAS"
cd "%projectdir%"


cd "%projectdir%packages\xtensor-blas-0.20.0"
cmake -S . -B %buildDir% -G "Ninja" -DCMAKE_INSTALL_PREFIX="%depdir%"
cmake --install %buildDir% --prefix "%depdir%\xtensor-blas"
cd "%projectdir%"
