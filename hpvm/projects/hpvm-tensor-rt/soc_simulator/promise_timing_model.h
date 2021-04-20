#include <iostream>
#include <fstream>
#include <string>

#include <cmath>
#include <cassert>
#include <algorithm>
#include <utility>
#include <vector>

// NOTE 1: This code uses seconds for all computations. Using clock cycles
// would have been cleaner but it's not possible because we don't know anything
// about Jetson's DRAM other than its bandwidth.

// NOTE 2: All elements are assumed to be 1 byte long.

// NOTE 3: PROMISE's frequency is fixed at 1 GHz and thus 1 cycle = 1 ns.

#define NUM_ARGS (7)
#define VOLTAGE_LEVELS (7)

class Dram {
private:
    const double energy_per_bit = 20e-12; // 20 pJ/bit
    double latency_;
    double bandwidth_;

public:
    Dram(const double latency, const double bandwidth);

    // Calculates (time, energy) of accessing 'num_bytes' in memory
    std::pair<double, double> access(const unsigned num_bytes, const bool pipeline = false) const;
};

class Scratchpad {
private:
    // Line size, latency, and energy
    const unsigned log_line_size = 6;
    const unsigned line_size     = 1 << log_line_size; // 64 B
    const double line_latency    = 1e-9;   // 1 ns
    const double line_energy     = 12e-12; // 12 pJ

    // Tag array. The tag is the address of the row being requested.
    unsigned num_lines_;
    std::vector<int> lines_;

    // DRAM
    Dram dram_;

    // Enable flag
    bool enable_;

private:
    // Calculates the index within the scratchpad array
    unsigned getIndex(const unsigned address) const;

public:
    Scratchpad(const bool enable,
               const unsigned size,
               const double dram_latency,
               const double dram_bandwidth);

    // Clears the scratchpad
    void clear();

    // Calculates (time, energy) of accessing 'num_bytes' starting from 'address'
    std::pair<double, double> access(const unsigned address,
                                     const unsigned num_bytes);
};

class Promise {
private:
    // Compute energy in pJ/128-element-dot-product for swings 0 through 7
    const double compute_energy_per_dot[VOLTAGE_LEVELS + 1] = {
        0.0, // This makes indexing simpler
        30.54403e-12,
        31.68943e-12,
        35.04211e-12,
        47.21840426e-12,
        52.68045671e-12,
        80.03489e-12,
        106.5494e-12
    };

    // SRAM access energy per byte
    const double sram_energy_per_byte = 0.1875e-12; // 0.1875 pJ/B

    // Leakage energy (converted from pJ/clock to mJ/s)
    const double leakage_energy_per_s   = 6e-3;   // 6 pJ/ns ==> 6 mJ/s

    const unsigned num_banks_ = 256;
    const unsigned bank_x_ = 128;
    const unsigned bank_y_ = 128;
	const unsigned bank_size = bank_x_ * bank_y_;
    const unsigned vector_size_ = bank_x_;

    const double pipeline_latency_ = 14e-9; // 14 ns
    const double reduction_latency_ = 10e-9; // 10 ns

    const bool use_scratchpad_ = false;
    const unsigned scratchpad_size_ = 512 * 1024; // 512 KB

    const double dram_latency_ = 100e-9; // 100 ns
    const double dram_bandwidth_ = 30e9;   // 30 GB/s (measured peak)

    // Scratchpad for array A
    Scratchpad scratch_;

    // DRAM
    Dram dram_;

    // uint version of min
    unsigned min(const unsigned x, const unsigned y) const;

    // Calculates energy of loading data into the SRAM
    double loadSRAM(const unsigned num_bytes) const;

    // Calculates (time, energy) of computing 'num_elements' elements
    std::pair<double, double> compute(const unsigned num_elements, 
                                      const unsigned voltage_swing) const;

    // Calculates the number of banks required to fill up an entire column; i.e. all the rows
    unsigned banksPerColumnTile(const unsigned num_rows) const;

    // Calculates the number of fully filled column tiles
    unsigned activeColumnTiles(const unsigned num_rows, const unsigned remaining_columns) const;

    // Calculates the number of rows of A that can be operated on in parallel
    // based on the tiling of *B*
    unsigned numRowsA(const unsigned num_rows, const unsigned num_cols) const;

    // Calculates (time, energy) of A x B (GEMM)
    std::pair<double, double> run(const unsigned num_rows_a,
                            const unsigned num_cols_a,
                            const unsigned num_rows_b,
                            const unsigned num_cols_b, 
                            const unsigned voltage_swing, 
                            const unsigned patch_factor);
public:
    Promise();

    std::pair<double, double> fc_profile(const unsigned num_rows_a,
                            const unsigned num_cols_a,
                            const unsigned num_rows_b,
                            const unsigned num_cols_b,
                            const unsigned voltage_swing,
                            const unsigned patch_factor);

    std::pair<double, double> conv_profile(const unsigned n,
                            const unsigned c,
                            const unsigned h,
                            const unsigned w,
                            const unsigned c_out,
                            const unsigned c_in,
                            const unsigned k_h,
                            const unsigned k_w,
                            const unsigned s_h,
                            const unsigned s_w,
                            const unsigned voltage_swing,
                            const unsigned patch_factor);
};
