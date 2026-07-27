# UnwrapMe

For bug reports, information and discussion go to Blenderartists.

https://blenderartists.org/t/1407021

# Build Instructions:
### Requirements:
Windows:
- Visual Studio 2022
- Python (same version as Blender)
- CMake
- Ninja

Linux:
- Clang 14
- Python (same version as Blender)
- CMake
- Ninja

### Steps
From a terminal run:
1. `cmake -P downloadPackages.cmake` to download dependencies (Warning: ~1GB).
2. `depinstall[.bat|.sh]` to build the dependencies.
3. `cmakeconfig[.bat|.sh]` to configure cmake.
4. `release build[.bat|.sh]` to build a release version.

`.bat` files are for Windows and `.sh` are for Linux.
