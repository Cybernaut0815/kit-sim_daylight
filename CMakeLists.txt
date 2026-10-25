cmake_minimum_required(VERSION 3.27)
project(daylight LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
# Own knob, not CMAKE_CUDA_ARCHITECTURES: project() (ours, or the server's before add_subdirectory) already cached nvcc's 52.
set(DAYLIGHT_CUDA_ARCHITECTURES "75-real;75-virtual" CACHE STRING "CUDA_ARCHITECTURES of src/gpu.cu (the OptiX-IR stays compute_75)")
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

find_package(CUDAToolkit 12.6 REQUIRED)

# OptiX headers from the installed SDK (NVIDIA proprietary, never committed or fetched).
set(OPTIX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0" CACHE PATH "OptiX SDK root (contains include/optix.h)")
set(OPTIX_INCLUDE "${OPTIX_INSTALL_DIR}/include")
if(NOT EXISTS "${OPTIX_INCLUDE}/optix.h")
  message(FATAL_ERROR "optix.h not found under ${OPTIX_INCLUDE}; set OPTIX_INSTALL_DIR")
endif()

# earcut.hpp (ISC): n-gon triangulation.
include(FetchContent)
FetchContent_Declare(earcut GIT_REPOSITORY https://github.com/mapbox/earcut.hpp GIT_TAG v2.2.4 GIT_SHALLOW TRUE SOURCE_SUBDIR include)
FetchContent_MakeAvailable(earcut)

# OptiX device program -> OptiX-IR -> embedded byte array. Only place with --use_fast_math (visibility only).
set(GEN "${CMAKE_CURRENT_BINARY_DIR}/gen")
file(MAKE_DIRECTORY "${GEN}")
add_custom_command(OUTPUT "${GEN}/trace.optixir"
  COMMAND "${CMAKE_CUDA_COMPILER}" -optix-ir -arch=compute_75 --use_fast_math -m64 -std=c++17
          -ccbin "${CMAKE_CXX_COMPILER}" -I "${OPTIX_INCLUDE}" -I "${CMAKE_CURRENT_SOURCE_DIR}/src"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/trace_dev.cu" -o "${GEN}/trace.optixir"
  DEPENDS src/trace_dev.cu src/params.h src/glass.h
  COMMENT "nvcc -optix-ir trace_dev.cu" VERBATIM)
add_custom_command(OUTPUT "${GEN}/trace_optixir.h"
  COMMAND "${CMAKE_COMMAND}" -DIN=${GEN}/trace.optixir -DOUT=${GEN}/trace_optixir.h -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake"
  DEPENDS "${GEN}/trace.optixir" cmake/embed.cmake
  COMMENT "embed trace.optixir" VERBATIM)

set(CXX_WARN /W4 /WX /permissive- /fp:precise /utf-8 /EHsc)

add_library(daylightcore STATIC src/mesh.cpp src/sky.cpp src/solar.cpp src/run.cpp src/cli.cpp src/gpu.cu "${GEN}/trace_optixir.h")
set_target_properties(daylightcore PROPERTIES CUDA_ARCHITECTURES "${DAYLIGHT_CUDA_ARCHITECTURES}")
target_include_directories(daylightcore PUBLIC src PRIVATE "${OPTIX_INCLUDE}" "${earcut_SOURCE_DIR}/include" "${GEN}")
target_link_libraries(daylightcore PUBLIC CUDA::cudart_static)
target_compile_definitions(daylightcore PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)
target_compile_options(daylightcore PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${CXX_WARN}>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>)   # ponytail: /W3 on the one nvcc host TU, CUDA headers trip /W4 /WX

add_executable(daylight_cli src/main.cpp)
set_target_properties(daylight_cli PROPERTIES OUTPUT_NAME daylight)
target_link_libraries(daylight_cli PRIVATE daylightcore)
target_compile_definitions(daylight_cli PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
target_compile_options(daylight_cli PRIVATE ${CXX_WARN})
target_link_options(daylight_cli PRIVATE /NODEFAULTLIB:libcmt)   # cudart_static pulls the static CRT; we are /MD

file(GLOB CHECK_SRC CONFIGURE_DEPENDS tests/check*.cpp)
add_executable(check ${CHECK_SRC})
target_link_libraries(check PRIVATE daylightcore)
target_compile_definitions(check PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
target_compile_options(check PRIVATE ${CXX_WARN})
target_link_options(check PRIVATE /NODEFAULTLIB:libcmt)

enable_testing()
add_test(NAME check COMMAND check "${CMAKE_CURRENT_SOURCE_DIR}/tests")
add_test(NAME cli_smoke COMMAND daylight_cli "${CMAKE_CURRENT_SOURCE_DIR}/tests/box.obj" --lat 41.98 --lon -87.92 --tz -6
         --quantity sunhours --time 06-21T12:00 --out "${CMAKE_CURRENT_BINARY_DIR}/smoke.txt")
add_test(NAME cli_smoke_npy COMMAND daylight_cli "${CMAKE_CURRENT_SOURCE_DIR}/tests/box.obj" --lat 41.98 --lon -87.92 --tz -6
         --quantity sunhours --time 06-21T12:00 --out "${CMAKE_CURRENT_BINARY_DIR}/smoke.npy")
add_test(NAME cli_smoke_clearsky COMMAND daylight_cli "${CMAKE_CURRENT_SOURCE_DIR}/tests/box.obj" --clear-sky --lat 41.98 --lon -87.92
         --quantity radiation --time 06-21T12:00 --out "${CMAKE_CURRENT_BINARY_DIR}/smoke_cs.txt")

option(DAYLIGHT_USD "USD input for the CLI via bindings/cpp (fetches LightUSD)" OFF)
if(DAYLIGHT_USD)
  add_subdirectory(bindings/cpp)
  target_compile_definitions(daylight_cli PRIVATE DAYLIGHT_USD)
  target_link_libraries(daylight_cli PRIVATE daylight_usd)   # carries the bindings/cpp include dir
  add_test(NAME cli_usd COMMAND daylight_cli "${CMAKE_CURRENT_SOURCE_DIR}/bindings/cpp/box.usda" --lat 41.98 --lon -87.92 --tz -6
           --quantity sunhours --time 06-21T12:00 --out "${CMAKE_CURRENT_BINARY_DIR}/usd_smoke.txt")
endif()
