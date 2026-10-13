// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
// CLI contract shared by main.cpp and other front ends (an HTTP server): flag parsing, the weather / time rules that
// follow parsing, and the output manifest. Return codes mirror the exit codes: 0 ok, 1 usage error, 2 runtime error.
#pragma once
#include "daylight.h"
#include <string>
#include <vector>

namespace daylight::cli {

extern const char* const kUsage;

struct Args {
  std::vector<std::string> sensors, context;   // positional mesh files / --context files
  Options opt;
  std::string epw, out, timeStr, fromStr, toStr;
  bool bench = false, clearSky = false, help = false;
  double turbidity = 3, altitude = 0;
  size_t benchSens = 1000000, benchTris = 10000000;
};

// argv[1..argc) -> a. 0 ok (-h/--help sets a.help and stops), 1 usage error with err set (the caller prints kUsage).
int parse(int argc, char** argv, Args& a, std::string& err);
// The rules that follow parsing: --clear-sky xor --epw; --clear-sky needs --lat --lon; tz defaults to round(lon / 15)
// when --lat --lon come without --tz; weather from a.epw or the clear-sky model unless haveWeather is already true on
// entry (a caller that parsed the EPW itself); --time / --from / --to parsed with the weather's leap flag into a.opt.
// 0 ok, 1 usage error, 2 runtime error (err set).
int prepare(Args& a, Weather& w, bool& haveWeather, std::string& err);
// The FILE.json manifest exactly as the CLI writes it; extraFields (e.g. "\"run_s\": 1.2") go before the closing brace.
std::string manifest_json(const Result& r, const std::string& extraFields = "");

}  // namespace daylight::cli
