#include "promise_timing_model.h"

// NOTE 1: This code uses seconds for all computations. Using clock cycles
// would have been cleaner but it's not possible because we don't know anything
// about Jetson's DRAM other than its bandwidth.

// NOTE 2: All elements are assumed to be 1 byte long.

// NOTE 3: PROMISE's frequency is fixed at 1 GHz and thus 1 cycle = 1 ns.

Dram::Dram(const double latency, const double bandwidth)
    : latency_(latency), bandwidth_(bandwidth) {}

// Calculates (time, energy) of accessing 'num_bytes' in memory
std::pair<double, double> Dram::access(const unsigned num_bytes, const bool pipeline) const{
    const auto time = (pipeline ? 0.0 : latency_) + (static_cast<double>(num_bytes)/bandwidth_);
    const auto energy = energy_per_bit * static_cast<double>(num_bytes * 8);
    return std::make_pair(time, energy);
}

// Calculates the index within the scratchpad array
unsigned Scratchpad::getIndex(const unsigned address) const {
    return ((address >> log_line_size) & (num_lines_ - 1));
}

Scratchpad::Scratchpad(const bool enable,
           const unsigned size,
           const double dram_latency,
           const double dram_bandwidth)
    : dram_(dram_latency, dram_bandwidth), enable_(enable) {

    num_lines_ = size / line_size;
    lines_.resize(num_lines_);
    clear();

#ifdef DEBUG
    if (enable_) {
        std::cout << "Initialized " << (size / 1024) << " KB scratchpad "
                  << "with geometry [" << num_lines_ << " x "
                  << line_size << "]\n";
    }
#endif
}

// Clears the scratchpad
void Scratchpad::clear() {
    for (auto &x : lines_)
        x = -1;
}

// Calculates (time, energy) of accessing 'num_bytes' starting from 'address'
std::pair<double, double> Scratchpad::access(const unsigned address,
                                 const unsigned num_bytes) {
    if (!enable_) {
        const auto load = dram_.access(num_bytes);
#ifdef DEBUG
        std::cout << "Accessing " << num_bytes << " bytes from DRAM\n";
        std::cout << "Took " << std::to_string(load.first * 1e6) << " us and "
                  << std::to_string(load.second * 1e6) << " uJ\n";
#endif
        return load;
    }

    auto addr = address;
    int num_bytes_remaining = static_cast<int>(num_bytes);

    double time = 0.0;
    double energy = 0.0;

    double hits = 0.0;
    double accesses = 0.0;

#ifdef DEBUG
    std::cout << "Accessing " << num_bytes << " bytes from the scratchpad, "
              << "starting at address " << addr << " (index "
              << getIndex(addr) << ")\n";
#endif

    // Keep reading line by line until everything is read
    while (num_bytes_remaining > 0) {
        if ((unsigned) lines_[getIndex(addr)] == address) {
            // Hit
            hits++;
        } else {
            // We missed. Load the line from memory. If this is not the
            // first miss, the accesses can be pipelined (overlapped).
            const bool first_miss = (hits == accesses);
            const auto miss = dram_.access(line_size, !first_miss);
            time += miss.first;
            energy += miss.second;

            lines_[getIndex(addr)] = address;
        }

        // This is required in case we began in the middle of a line
        const auto bytes_accessed = line_size - (addr & (line_size - 1));
        addr += bytes_accessed;
        num_bytes_remaining -= bytes_accessed;

        time += line_latency;
        energy += line_energy;
        accesses++;
    }

#ifdef DEBUG
    std::cout << "Took " << std::to_string(time * 1e6) << " us and "
              << std::to_string(energy * 1e6) << " uJ\n";
    std::cout << "Hit rate is " << ((hits * 100.0) / accesses) << "%\n";
#endif
    return std::make_pair(time, energy);
}

// uint version of min
unsigned Promise::min(const unsigned x, const unsigned y) const {
    return static_cast<unsigned>(std::min(x, y));
}

// Calculates energy of loading data into the SRAM
double Promise::loadSRAM(const unsigned num_bytes) const {
    return (sram_energy_per_byte * static_cast<double>(num_bytes));
}

// Calculates (time, energy) of computing 'num_elements' elements
std::pair<double, double> Promise::compute(const unsigned num_elements, 
                                  const unsigned voltage_swing) const {
    const auto time = (pipeline_latency_ * static_cast<double>(num_elements)) + reduction_latency_;
    const auto energy = compute_energy_per_dot[voltage_swing] * static_cast<double>(num_elements);
    return std::make_pair(time, energy);
}

// Calculates the number of banks required to fill up an entire column; i.e. all the rows
unsigned Promise::banksPerColumnTile(const unsigned num_rows) const {
    return static_cast<unsigned>(std::ceil(static_cast<double>(num_rows) / static_cast<double>(bank_x_)));
}

// Calculates the number of fully filled column tiles
unsigned Promise::activeColumnTiles(const unsigned num_rows, const unsigned remaining_columns) const {
    const auto banks_per_column_tile = banksPerColumnTile(num_rows);
    const auto remaining_column_tiles = static_cast<unsigned>(std::ceil(static_cast<double>(remaining_columns) / static_cast<double>(bank_y_)));
    auto active_column_tiles = num_banks_ / banks_per_column_tile;
    active_column_tiles = min(active_column_tiles, remaining_column_tiles);
    return active_column_tiles;
}

// Calculates the number of rows of A that can be operated on in parallel
// based on the tiling of *B*
unsigned Promise::numRowsA(const unsigned num_rows, const unsigned num_cols) const {
    const auto banks_per_column_tile = banksPerColumnTile(num_rows);
    const auto total_column_tiles = static_cast<unsigned>(std::ceil(static_cast<double>(num_cols) / static_cast<double>(bank_y_)));
    const auto total_required_banks = banks_per_column_tile * total_column_tiles;
    const auto num_rows_a = num_banks_ < total_required_banks ? 1 : num_banks_ / total_required_banks;
    return num_rows_a;
}

// Calculates (time, energy) of A x B (GEMM)
std::pair<double, double> Promise::run(const unsigned num_rows_a,
                        const unsigned num_cols_a,
                        const unsigned num_rows_b,
                        const unsigned num_cols_b, 
                        const unsigned voltage_swing, 
                        const unsigned patch_factor) {
#ifdef DEBUG
    std::cout << "Performing [" << num_rows_a << " x " << num_cols_a
              << "] x [" << num_rows_b << " x " << num_cols_b << "] GEMM\n";
#endif
    scratch_.clear();

    double compute_time = 0.0;
    double compute_energy = 0.0;

    double leakage_energy = 0.0;

    double a_time = 0.0;
    double a_energy = 0.0;

    double b_time = 0.0;
    double b_energy = 0.0;

    double c_time = 0.0;
    double c_energy = 0.0;

    double average_bank_utilization;
    double iterations;

    // Load a tile of B, compute the corresponding part of C, repeat
    auto remaining_columns_b = num_cols_b;
    for (unsigned i = 0; i < num_cols_b;) {
        // Figure out how B is tiled. In a nutshell, we use as many banks
        // as will fill up entire columns of B (because we need an entire
        // column for the reduction to work). The corner cases are where
        // either #rows or #columns is not divisible by the bank size,
        // and/or the banks only fill up part of the column. Once the
        // tiling and #active banks is figured out, we can calculate the
        // tile size.
        // Furthermore, if B is sufficiently small, we may be able to
        // operate on multiple rows of A at the same time.
        const auto banks_per_column_tile = banksPerColumnTile(num_rows_b);
        const auto active_column_tiles = activeColumnTiles(num_rows_b, remaining_columns_b);
        const auto tile_x = min(num_rows_b, banks_per_column_tile * bank_x_);
        const auto tile_y = min(remaining_columns_b, active_column_tiles * bank_y_);
        const auto max_parallel_rows_a = min(num_rows_a, numRowsA(num_rows_b, num_cols_b));
        const auto max_active_banks = banks_per_column_tile * active_column_tiles * max_parallel_rows_a;

        // Load the required tiles of B into the active banks
        const auto num_bytes = (tile_x * tile_y) / patch_factor;
        const auto load_b = dram_.access(num_bytes);
        b_time += load_b.first;
        b_energy += load_b.second;
        b_energy += loadSRAM(num_bytes);
        leakage_energy += (load_b.first * leakage_energy_per_s * max_active_banks);

#ifdef DEBUG
        std::cout << "\nLoading " << tile_x << " x " << tile_y << " tile of B from DRAM\n";
        std::cout << "There are " << active_column_tiles << " active column tiles of B "
                  << "with " << banks_per_column_tile << " PROMISE banks per tile\n";
#endif

        // Load row(s) of A, compute C, write the result back
        auto remaining_rows_a = num_rows_a;
        for (unsigned j = 0; j < num_rows_a; j += max_parallel_rows_a) {
            const auto active_rows_a = min(remaining_rows_a, max_parallel_rows_a);
            const auto active_banks = banks_per_column_tile * active_column_tiles * active_rows_a;
            const auto bank_utilization = (static_cast<double>(active_banks) * 100.0) / static_cast<double>(num_banks_);
            average_bank_utilization += bank_utilization;
            iterations++;

#ifdef DEBUG
            std::cout << "There are a total of " << active_banks << " active banks "
                      << "operating on " << active_rows_a << " rows of A in parallel\n";
            std::cout << "Bank utilization is " << bank_utilization << "%\n";
#endif

            // Load the rows from the scratchpad
            for (unsigned k = 0; k < active_rows_a; k++) {
                const auto load_a = scratch_.access((j + k) * num_cols_a, num_cols_a);
                a_time += load_a.first;
                a_energy += load_a.second;
                leakage_energy += (load_a.first * leakage_energy_per_s * active_banks);
            }

            // All the banks operate in parallel, so use the biggest
            // computation and count the time only once. Computation
            // energy is energy per bank x active banks.
            const auto comp_c = compute(tile_y > bank_y_ ? bank_y_ : tile_y, voltage_swing);
            compute_time += comp_c.first;
            compute_energy += (comp_c.second * active_banks);

            // This is sequential, so use tile width and the number of active rows
            const auto store_c = dram_.access(tile_y * active_rows_a);
            c_time += store_c.first;
            c_energy += store_c.second;

            // Leakage is for the entire duration and across all active banks
            leakage_energy += ((comp_c.first + store_c.first) * leakage_energy_per_s * active_banks);

            remaining_rows_a -= active_rows_a;
        }

        auto processed_columns_b = active_column_tiles * bank_y_;
        i += processed_columns_b;
        remaining_columns_b -= processed_columns_b;
    }

    const auto memory_time = a_time + b_time + c_time;
    const auto memory_energy = a_energy + b_energy + c_energy;
    const auto total_time = compute_time + memory_time;
    const auto total_energy = compute_energy + memory_energy + leakage_energy;

#ifdef DEBUG
    std::cout << "------------------------------\n";
    std::cout << "Compute time:   " << std::to_string(compute_time * 1e3) << " ms\n";
    std::cout << "Compute energy: " << std::to_string(compute_energy * 1e3) << " mJ\n";
    std::cout << "Compute power:  " << std::to_string((compute_energy/compute_time) * 1e3) << " mW\n";
    std::cout << "------------------------------\n";

    std::cout << "Memory time:   " << std::to_string(memory_time * 1e3) << " ms\n";
    std::cout << "          A:   " << std::to_string(a_time * 1e3) << " ms\n";
    std::cout << "          B:   " << std::to_string(b_time * 1e3) << " ms\n";
    std::cout << "          C:   " << std::to_string(c_time * 1e3) << " ms\n";
    std::cout << "Memory energy: " << std::to_string(memory_energy * 1e3) << " mJ\n";
    std::cout << "            A: " << std::to_string(a_energy * 1e3) << " mJ\n";
    std::cout << "            B: " << std::to_string(b_energy * 1e3) << " mJ\n";
    std::cout << "            C: " << std::to_string(c_energy * 1e3) << " mJ\n";
    std::cout << "Memory power:  " << std::to_string((memory_energy/memory_time) * 1e3) << " mW\n";
    std::cout << "------------------------------\n";

    std::cout << "Leakage energy: " << std::to_string(leakage_energy * 1e3) << " mJ\n";
    std::cout << "Leakage power:  " << std::to_string((leakage_energy/total_time) * 1e3) << " mW\n";
    std::cout << "------------------------------\n";

    std::cout << "Total time:    " << std::to_string(total_time * 1e3) << " ms\n";
    std::cout << "Total energy:  " << std::to_string(total_energy * 1e3) << " mJ\n";
    std::cout << "Average power: " << std::to_string((total_energy/total_time) * 1e3) << " mW\n";
    std::cout << "------------------------------\n";

    std::cout << "Average bank utilization was " << (average_bank_utilization / iterations) << "%\n";
    std::cout << "------------------------------\n";
#endif

    //std::vector<double> result = {total_time, total_energy, compute_time, compute_energy, memory_time, memory_energy, leakage_energy};
    //return result;
    return std::make_pair(total_time, total_energy);
}

Promise::Promise() : 
        scratch_(use_scratchpad_, scratchpad_size_, dram_latency_, dram_bandwidth_),
        dram_(dram_latency_, dram_bandwidth_) {
#ifdef DEBUG
    std::cout << "Initialized PROMISE with " << num_banks_ << " ["
              << bank_x << " x " << bank_y << "] banks\n";
#endif
}

// TODO better naming?
std::pair<double, double> Promise::fc_profile(const unsigned num_rows_a,
                        const unsigned num_cols_a,
                        const unsigned num_rows_b,
                        const unsigned num_cols_b,
                        const unsigned voltage_swing,
                        const unsigned patch_factor) {
    return num_rows_a <= num_cols_b ?
            run(num_rows_a, num_cols_a, num_rows_b, num_cols_b, voltage_swing, patch_factor) :
            run(num_cols_b, num_rows_b, num_cols_a, num_rows_a, voltage_swing, patch_factor);
}

std::pair<double, double> Promise::conv_profile(const unsigned n,
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
                        const unsigned patch_factor) { 
    unsigned num_rows_a = n * h * w / (s_h * s_w);
    unsigned num_cols_a = c_in * k_h * k_w;
    unsigned num_rows_b = num_rows_a;
    unsigned num_cols_b = c_out;

    return num_rows_a <= num_cols_b ? 
            run(num_rows_a, num_cols_a, num_rows_b, num_cols_b, voltage_swing, patch_factor) :
            run(num_cols_b, num_rows_b, num_cols_a, num_rows_a, voltage_swing, patch_factor);
}

/*
int main(int argc, char *argv[]) {
    if (argc != NUM_ARGS) {
        std::cout << "Usage: " << argv[0] << " <#rows A> <#cols A> <#rows B> <#cols B> <patch factor> <voltage swing>\n";
        exit(1);
    }

    // Inputs
    const auto num_rows_a = std::atoi(argv[1]);
    const auto num_cols_a = std::atoi(argv[2]);
    const auto num_rows_b = std::atoi(argv[3]);
    const auto num_cols_b = std::atoi(argv[4]);
    const auto patch_factor = std::atoi(argv[5]);
    const auto voltage_swing = std::atoi(argv[6]);

    // Make sure the array dimensions make sense and the swing level is valid
    assert(num_cols_a == num_rows_b);
    assert(voltage_swing > 0 and voltage_swing <= VOLTAGE_LEVELS);

    Promise promise;

    auto result = promise.fc_profile(num_rows_a, num_cols_a, num_rows_b, num_cols_b, voltage_swing, patch_factor);
    std::cout << std::to_string(result.first * 1e3) << ","
              << std::to_string(result.second * 1e3) << std::endl;
    return 0;
}
*/
