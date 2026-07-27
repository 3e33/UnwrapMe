buildDir=build/linux

cmake -S . -B $buildDir -G "Ninja Multi-Config" -DCMAKE_C_COMPILER="clang" -DCMAKE_CXX_COMPILER="clang++" $1
