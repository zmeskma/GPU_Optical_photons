@echo off
rem ---------------------------------------------------------------------------
rem Build the CUDA configuration on Windows without a system-wide CUDA toolkit.
rem
rem Uses the CUDA 12.2 compiler from conda-forge in a self-contained folder
rem (no admin rights), created once with the standalone micromamba:
rem
rem   micromamba create -p C:\Users\%USERNAME%\cuda-12.2-conda\env -c conda-forge ^
rem       cuda-version=12.2 cuda-nvcc cuda-cudart-dev cmake ninja
rem
rem and Visual Studio's MSVC as the host compiler. CUDA 12.2 predates the
rem newest MSVC toolsets, hence -allow-unsupported-compiler and the STL
rem version-check override below (pick the oldest toolset you have).
rem
rem usage: scripts\build_windows_cuda.bat [build-dir] [extra cmake args...]
rem ---------------------------------------------------------------------------
setlocal
if "%CUDA_CONDA_ENV%"=="" set "CUDA_CONDA_ENV=%USERPROFILE%\cuda-12.2-conda\env"
if "%VCVARS%"=="" set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if "%VCVARS_VER%"=="" set "VCVARS_VER=14.44"
if "%CUDA_ARCH%"=="" set "CUDA_ARCH=75"
set "BUILD_DIR=%~1"
if "%BUILD_DIR%"=="" set "BUILD_DIR=build-cuda-win"

call "%VCVARS%" -vcvars_ver=%VCVARS_VER% >nul || exit /b 1
set "PATH=%CUDA_CONDA_ENV%\Library\bin;%CUDA_CONDA_ENV%\Library\nvvm\bin;%PATH%"
set "CUDA_PATH=%CUDA_CONDA_ENV%\Library"

cmake -S "%~dp0.." -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DWITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCH% ^
  -DCMAKE_CUDA_COMPILER="%CUDA_CONDA_ENV%\Library\bin\nvcc.exe" ^
  -DCUDAToolkit_ROOT="%CUDA_CONDA_ENV%\Library" ^
  "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH" ^
  %EXTRA_CMAKE_ARGS% || exit /b 1
cmake --build "%BUILD_DIR%" || exit /b 1
endlocal
