// SPDX-License-Identifier: MIT
// USD stage -> daylight::Mesh[] (LightUSD). Optional binding, not part of the core (bindings/cpp/CMakeLists.txt).
#pragma once
#include "daylight.h"
#include <string>
#include <vector>

namespace daylight {
// upAxis: 'A' = stage metadata (fallback Y), 'Y', 'Z'. metersPerUnit <= 0 = authored value (USD fallback 0.01).
bool read_usd(const std::string& path, char upAxis, double metersPerUnit, std::vector<Mesh>& out, std::string& err);
}  // namespace daylight
