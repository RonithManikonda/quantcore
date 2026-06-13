#pragma once

// Shared helper for path-batch parallelism. Several simulators split work the
// same way — one contiguous batch of paths per thread — so the thread-count
// policy lives here in one place.

#include <algorithm>
#include <thread>

// Resolve a usable worker-thread count:
//   n_threads <= 0  -> auto-detect via std::thread::hardware_concurrency()
//   still 0         -> fall back to 4 (some platforms report 0)
//   capped at n_items so we never spawn threads with no work to do
inline int resolve_n_threads(int n_threads, int n_items) {
    if (n_threads <= 0)
        n_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (n_threads <= 0)
        n_threads = 4;
    return std::min(n_threads, n_items);
}
