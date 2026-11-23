cmake_minimum_required(VERSION 3.24)

option(FPG_CPU_ONLY "Build without CUDA (the tests and --cpu runs still work)" OFF)

if(FPG_CPU_ONLY)
  project(forza_painter_cuda LANGUAGES CXX)
else()
  # sm_75 (RTX 20), sm_86 (RTX 30, the 3090), sm_89 (RTX 40) as machine code,
  # plus sm_90 machine code and PTX so newer cards can compile it at load time.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75-real 86-real 89-real 90)
  endif()
  project(forza_painter_cuda LANGUAGES CXX CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # Link the CUDA runtime into the EXE: users need only their normal driver.
  set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(MSVC)
  # Static C runtime too, so the EXE runs without the VC++ redistributable.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

add_library(fpg_core STATIC
  src/settings.cpp
  src/platform.cpp
  src/image_io.cpp
  src/geometry_json.cpp
  src/tables.cpp
  src/backend_cpu.cpp
  src/engine.cpp
  src/tools.cpp)
target_include_directories(fpg_core PUBLIC src third_party/stb)

# The CPU and GPU must round the same way, so multiply-add fusing is off on
# both (MSVC does not fuse under its default /fp:precise).
if(MSVC)
  target_compile_options(fpg_core PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/fp:precise /utf-8 /W3>)
  target_compile_definitions(fpg_core PUBLIC _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(fpg_core PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off -Wall -Wextra>)
endif()

if(NOT FPG_CPU_ONLY)
  target_sources(fpg_core PRIVATE src/gpu/backend_cuda.cu)
  target_compile_definitions(fpg_core PUBLIC FPG_WITH_CUDA)
  target_compile_options(fpg_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false -lineinfo>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/fp:precise,/utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Xcompiler=-ffp-contract=off>)
endif()

add_executable(forza-painter-geometrize-cuda src/main.cpp)
target_link_libraries(forza-painter-geometrize-cuda PRIVATE fpg_core)

enable_testing()
add_executable(fpg_tests tests/test_main.cpp)
target_include_directories(fpg_tests PRIVATE tests)
target_link_libraries(fpg_tests PRIVATE fpg_core)
add_test(NAME fpg_tests
         COMMAND fpg_tests ${CMAKE_CURRENT_SOURCE_DIR}/tests/data ${CMAKE_CURRENT_BINARY_DIR}/test-output)
