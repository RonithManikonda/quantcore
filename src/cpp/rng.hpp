#pragma once

// All RNG utilities are here so brownian.cpp stays focused on math.
//
// C++ standard library headers you will use in brownian.cpp:
//   <random>  — std::mt19937_64, std::normal_distribution<double>
//   <cstdint> — uint64_t

#include <random>
#include <cstdint>

// Returns a seeded Mersenne Twister (64-bit) RNG.
// Use this in simulate_paths / simulate_increments so that the same seed
// always produces the same sequence of random numbers.
inline std::mt19937_64 make_seeded_rng(uint64_t seed) {
    return std::mt19937_64(seed);
}

// Derives a per-thread RNG from a base seed and a thread index.
// Used in Phase 6: each thread gets its own independent RNG stream so there
// is no contention and results are still deterministic for a given seed.
//
// Seeding strategy: XOR the base seed with (thread_id * large_odd_constant).
// This is simple and avoids correlation between streams for small thread counts.
// If you want stronger independence, look up "jump-ahead" for Mersenne Twister.
inline std::mt19937_64 make_thread_rng(uint64_t base_seed, int thread_id) {
    // 6364136223846793005 is the LCG multiplier from Knuth — large and odd.
    const uint64_t salt = static_cast<uint64_t>(thread_id) * 6364136223846793005ULL;
    return std::mt19937_64(base_seed ^ salt);
}
