// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
// Assert-style checks, run by ctest. Each module owns tests/check_<module>.cpp; each returns its failure count.
#include <cstdio>
int check_sky(const char* testsDir);
int check_solar(const char* testsDir);
int check_mesh(const char* testsDir);
int check_gpu(const char* testsDir);
int main(int argc, char** argv) {
  const char* dir = argc > 1 ? argv[1] : "tests";
  int failures = 0;
  failures += check_sky(dir);
  failures += check_solar(dir);
  failures += check_mesh(dir);
  failures += check_gpu(dir);
  std::printf("%s: %d failure(s)\n", failures ? "FAIL" : "OK", failures);
  return failures ? 1 : 0;
}
