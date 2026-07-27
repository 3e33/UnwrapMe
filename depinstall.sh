projectdir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
depdir=$projectdir/dep/linux
buildDir=build/linux

echo "This might take a while, continue?"
select answer in "Yes" "No"; do
    case $answer in
        Yes) break;;
        *) exit 0;;
    esac
done

cd "$projectdir/packages/pybind11-2.10.3"
cmake -S . -B $buildDir -G "Ninja"
cmake --install $buildDir --prefix "$depdir/pybind11"
cd "$projectdir"


cd "$projectdir/packages/xtl-0.7.5"
cmake -S . -B $buildDir -G "Ninja" -DCMAKE_INSTALL_PREFIX="$depdir"
cmake --install $buildDir --prefix "$depdir/xtl"
cd "$projectdir"


cd "$projectdir/packages/xtensor-0.24.4"
cmake -S . -B $buildDir -G "Ninja" -DCMAKE_INSTALL_PREFIX="$depdir"
cmake --install $buildDir --prefix "$depdir/xtensor"
cd "$projectdir"


cd "$projectdir/packages/xsimd-10.0.0"
cmake -S . -B $buildDir -G "Ninja" -DCMAKE_INSTALL_PREFIX="$depdir"
cmake --install $buildDir --prefix "$depdir/xsimd"
cd "$projectdir"


cd "$projectdir/packages/xtensor-python-0.26.1"
cmake -S . -B $buildDir -G "Ninja" -DCMAKE_INSTALL_PREFIX="$depdir"
cmake --install $buildDir --prefix "$depdir/xtensor-python"
cd "$projectdir"


cd "$projectdir/packages/OpenBLAS-0.3.21"
cmake -S . -B $buildDir -G "Ninja" \
 -DCMAKE_CXX_COMPILER=clang++ \
 -DCMAKE_C_COMPILER=clang \
 -DDYNAMIC_ARCH=OFF \
 -DBUILD_SHARED_LIBS=OFF \
 -DCMAKE_BUILD_TYPE=Release \
 -DBUILD_TESTING=OFF \
 -DBUILD_STATIC_LIBS=ON \
 -DNOFORTRAN=ON \
 -DTARGET=NEHALEM
cmake --build $buildDir --config Release
cmake --install $buildDir --prefix "$depdir/OpenBLAS"
cd "$projectdir"

cd "$projectdir/packages/xtensor-blas-0.20.0"
cmake -S . -B $buildDir -G "Ninja" -DCMAKE_INSTALL_PREFIX="$depdir"
cmake --install $buildDir --prefix "$depdir/xtensor-blas"
cd "$projectdir"
