# bindings/cpp: LightUSD-based USD reader (lib daylight_usd, linked into the CLI by the root) + check_usd test.
# Built only with -DDAYLIGHT_USD=ON (fetches LightUSD). The core library never sees USD.
include(FetchContent)
# LightUSD (Apache-2.0, formerly TinyUSDZ). Heavy optional features off.
foreach(opt TEXTOOLS EXR TIFF AUDIO USDMTLX JSON USD_TO_GLTF USDVOX USDVOL COLORIO NANOIMAGE BUILTIN_IMAGE_LOADER)
  set(LIGHTUSD_WITH_${opt} OFF CACHE BOOL "" FORCE)
endforeach()
set(LIGHTUSD_WITH_TYDRA ON CACHE BOOL "" FORCE)
set(LIGHTUSD_WITH_PCP ON CACHE BOOL "" FORCE)
set(LIGHTUSD_ENABLE_NATIVE_SIMD OFF CACHE BOOL "" FORCE)
set(LIGHTUSD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LIGHTUSD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(lightusd GIT_REPOSITORY https://github.com/lighttransport/LightUSD GIT_TAG v1.0.0-rc4 GIT_SHALLOW TRUE GIT_PROGRESS FALSE)
FetchContent_MakeAvailable(lightusd)

add_library(daylight_usd STATIC daylight_usd.cpp)
target_include_directories(daylight_usd PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" PRIVATE "${lightusd_SOURCE_DIR}/src")
target_link_libraries(daylight_usd PUBLIC daylightcore PRIVATE lightusd::lightusd_static)
add_executable(check_usd check_usd.cpp)
foreach(t daylight_usd check_usd)
  target_compile_options(${t} PRIVATE ${CXX_WARN})
  target_compile_definitions(${t} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)
endforeach()
target_link_libraries(check_usd PRIVATE daylight_usd)
target_link_options(check_usd PRIVATE /NODEFAULTLIB:libcmt)   # cudart_static pulls the static CRT; we are /MD
add_test(NAME check_usd COMMAND check_usd "${CMAKE_CURRENT_SOURCE_DIR}")
