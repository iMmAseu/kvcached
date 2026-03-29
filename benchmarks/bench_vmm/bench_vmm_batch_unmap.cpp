// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <ratio>
#include <sstream>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_utils.hpp"

namespace {

struct Config {
  std::vector<int> pages_list;
  int repeats = 20;
  int warmup_repeats = 3;
  size_t page_size = 2ul << 20; // 2MB
};

std::string trim(const std::string &s) {
  const auto begin = s.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos)
    return "";
  const auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(begin, end - begin + 1);
}

std::vector<int> parse_pages_list(const std::string &raw) {
  std::vector<int> values;
  std::stringstream ss(raw);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token = trim(token);
    if (token.empty())
      continue;
    const int v = std::stoi(token);
    if (v > 0) {
      values.push_back(v);
    }
  }
  if (values.empty()) {
    throw std::runtime_error("pages list is empty");
  }
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

Config parse_args(int argc, char **argv) {
  Config cfg;
  cfg.pages_list =
      parse_pages_list("1,2,4,8,12,16,20,24,32,48,64");

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };

    if (arg == "--pages-list") {
      cfg.pages_list = parse_pages_list(need_value("--pages-list"));
    } else if (arg == "--repeats") {
      cfg.repeats = std::stoi(need_value("--repeats"));
    } else if (arg == "--warmup-repeats") {
      cfg.warmup_repeats = std::stoi(need_value("--warmup-repeats"));
    } else if (arg == "--page-size-mb") {
      const int mb = std::stoi(need_value("--page-size-mb"));
      if (mb <= 0) {
        throw std::runtime_error("--page-size-mb must be > 0");
      }
      cfg.page_size = static_cast<size_t>(mb) * (1ul << 20);
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: bench_vmm_batch_unmap.bin [options]\n"
          << "  --pages-list 1,2,4,...       Comma-separated pages per unmap call\n"
          << "  --repeats N                  Timed repeats per pages count\n"
          << "  --warmup-repeats N           Warmup repeats per pages count\n"
          << "  --page-size-mb N             CUDA VMM page size in MB (default 2)\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (cfg.repeats <= 0) {
    throw std::runtime_error("--repeats must be > 0");
  }
  if (cfg.warmup_repeats < 0) {
    throw std::runtime_error("--warmup-repeats must be >= 0");
  }
  return cfg;
}

void init_cuda() {
  CHECK_RT(cudaFree(0));
  CHECK_DRV(cuInit(0));

  CUcontext ctx;
  CHECK_DRV(cuDevicePrimaryCtxRetain(&ctx, 0));
  CHECK_DRV(cuCtxSetCurrent(ctx));
}

CUmemAllocationProp make_alloc_prop() {
  CUdevice dev;
  CHECK_DRV(cuCtxGetDevice(&dev));
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = dev;
  return prop;
}

CUmemAccessDesc make_access_desc() {
  CUdevice dev;
  CHECK_DRV(cuCtxGetDevice(&dev));
  CUmemAccessDesc access = {};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = dev;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  return access;
}

double percentile_ms(std::vector<double> values, double q) {
  if (values.empty())
    return 0.0;
  std::sort(values.begin(), values.end());
  const double idx = (values.size() - 1) * (q / 100.0);
  const auto lo = static_cast<size_t>(std::floor(idx));
  const auto hi = std::min(lo + 1, values.size() - 1);
  const double frac = idx - static_cast<double>(lo);
  return values[lo] * (1.0 - frac) + values[hi] * frac;
}

double run_one_batch_unmap(int pages, size_t page_size,
                           const CUmemAllocationProp &prop,
                           const CUmemAccessDesc &access_desc) {
  const size_t bytes = static_cast<size_t>(pages) * page_size;
  CUdeviceptr addr = 0;
  CHECK_DRV(cuMemAddressReserve(&addr, bytes, page_size, 0, 0));

  std::vector<CUmemGenericAllocationHandle> handles(pages);
  for (int i = 0; i < pages; ++i) {
    CHECK_DRV(cuMemCreate(&handles[i], page_size, &prop, 0));
    CHECK_DRV(cuMemMap(addr + static_cast<CUdeviceptr>(i) * page_size, page_size,
                       0, handles[i], 0));
    CHECK_DRV(cuMemSetAccess(addr + static_cast<CUdeviceptr>(i) * page_size,
                             page_size, &access_desc, 1));
  }

  CHECK_RT(cudaDeviceSynchronize());
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < pages; ++i) {
    CHECK_DRV(
        cuMemUnmap(addr + static_cast<CUdeviceptr>(i) * page_size, page_size));
  }
  CHECK_RT(cudaDeviceSynchronize());
  const auto t1 = std::chrono::high_resolution_clock::now();

  for (auto handle : handles) {
    CHECK_DRV(cuMemRelease(handle));
  }
  CHECK_DRV(cuMemAddressFree(addr, bytes));

  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Config cfg = parse_args(argc, argv);
    init_cuda();

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CHECK_DRV(cuMemGetInfo(&free_bytes, &total_bytes));

    std::cout << "[setup] pure CUDA VMM batch unmap benchmark" << std::endl;
    std::cout << "[setup] page_size=" << (cfg.page_size >> 20)
              << "MB repeats=" << cfg.repeats
              << " warmup_repeats=" << cfg.warmup_repeats << std::endl;
    std::cout << "[setup] free_memory_gb="
              << std::fixed << std::setprecision(2)
              << (static_cast<double>(free_bytes) / std::giga::num) << std::endl;

    const auto prop = make_alloc_prop();
    const auto access_desc = make_access_desc();

    for (const int pages : cfg.pages_list) {
      std::vector<double> latencies_ms;
      latencies_ms.reserve(static_cast<size_t>(cfg.repeats));
      const int total_runs = cfg.warmup_repeats + cfg.repeats;
      for (int rep = 0; rep < total_runs; ++rep) {
        const double elapsed_ms =
            run_one_batch_unmap(pages, cfg.page_size, prop, access_desc);
        if (rep >= cfg.warmup_repeats) {
          latencies_ms.push_back(elapsed_ms);
        }
      }

      const double mean_ms =
          std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) /
          static_cast<double>(latencies_ms.size());
      const double p95_ms = percentile_ms(latencies_ms, 95.0);
      const double per_page_ms = mean_ms / static_cast<double>(pages);
      const double batch_unmap_mb =
          static_cast<double>(pages) * static_cast<double>(cfg.page_size) /
          (1024.0 * 1024.0);

      std::cout << "[result] pages=" << std::setw(3) << pages
                << " mean_unmapped=" << std::fixed << std::setprecision(2)
                << static_cast<double>(pages) << "/" << pages
                << " mean_unmap=" << std::setprecision(4) << mean_ms << "ms"
                << " p95=" << p95_ms << "ms"
                << " per_page=" << per_page_ms << "ms"
                << " batch_unmap_bytes=" << std::setprecision(2)
                << batch_unmap_mb << "MB" << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "[error] " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
