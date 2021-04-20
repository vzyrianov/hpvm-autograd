#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// Reads power rails at runtime and computes the specified device and DDR energy within a window  
// of time, which is delimitered by the calls to resume_profiler() and pause_profiler()
// 
// IMPORTANT: Must call pause_profiler() to kill the profiler thread 
//
// Public interface methods:
//      void start_profiler();
//      void resume_profiler(); 
//      void pause_profiler(); 
//      std::pair<double, double> get_time_energy() const;
//      void reset() 
//      void pause_profiler();
class Profiler {
public:
    using clock_type = std::chrono::high_resolution_clock;

    // FIXME: Probably change the enum name to something else.
    // Expose the devices available
    enum Device {
        CPU = 0,
        GPU = 1 << 0,
        SOC = 1 << 1,
        SYS = 1 << 2,
        NONE = 1 << 3,
    };
    
    Profiler(const Device dev = GPU);
    
    ~Profiler();

    // Reinitializes boolean vars used for control flow and launches the profiler 
    // thread. DOES NOT reset other internal data structures. 
	void start_profiler();

    // Resumes the profiling of whatever executable's currently running
    // DOES NOT reset any data 
    void resume_profiler();

    // Stops profiler by putting profiler thread to sleep 
	void pause_profiler();

    // Gets the delta time and total CPU/GPU and DDR energy between the last two
    // calls to resume_profiler and pause_profiler
    //
    // Returns this as a pair of <delta time in milliseconds, energy>
	//std::pair<double, double> get_time_energy(const Device dev) const;
    std::pair<double, double> get_time_energy() const;

    // Resets all internal data structures, including the vector storing all power_readings.
	void reset();

    // Exit the profiler and kill the thread
    // Must call start_profiler() to reuse this object after calling pause_profiler()
    void stop_profiler();

private:
    // Jetson's ARM cores' physical IDs. The two Denver cores are 1 and 2, and
    // we can't use them.
    const unsigned core0 = 0;
    const unsigned core1 = 3;
    const unsigned core2 = 4;
    const unsigned core3 = 5;

    // Power rails are mounted as files. Keeping the old power rail file names for possible future
    // integrations
    const std::string cpu_power_rail = "/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_power1_input";
    const std::string gpu_power_rail = "/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_power0_input";
    const std::string ddr_power_rail = "/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_power2_input";
    const std::string soc_power_rail = "/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_power1_input";
    const std::string sys_power_rail = "/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_power0_input";
    // Critical assumption: If this file doesn't exist, then the board isn't a Jetson
    const std::string jetson_chip_id = "/sys/module/tegra_fuse/parameters/tegra_chip_id";

    // True if running on Jetson, else false
    bool on_jetson_; 

    // An individual power reading
    struct PowerReading {
        std::chrono::time_point<clock_type> time_;
        double dev_;
        double ddr_;
        
        //Device dev_type_;
       
       // Initialize to avoid any undefined behavior
       //PowerReading() : dev_type_(NONE), dev_(0), ddr_(0) {}
    };

    // Device the profiler is meant to be used for.
    Device prof_dev;

    // Stores all power readings and is cleared only when reset() is called
    std::vector<PowerReading> power_readings_;

    // For reading the i2c buses via sysfs
    std::ifstream cpu_stream_;
    std::ifstream gpu_stream_;
    std::ifstream ddr_stream_;
    std::ifstream soc_stream_;
    std::ifstream sys_stream_;

    double run_profiler_overhead = 0.0;

    mutable std::mutex vector_mutex_;

    std::mutex mutex_;
    
    std::condition_variable cond_var_;

    std::chrono::time_point<clock_type> start_time_;

    std::chrono::time_point<clock_type> stop_time_;

    std::atomic_bool should_run_profiler_; // True if we want to resume the profiling thread

    std::atomic_bool should_stop_profiler_; // Quit profiling

    std::thread profiler_thread_;

    // Obtain's a single power reading from the device and DDR rails
    //void obtain_power_reading(Device dev);
	void obtain_power_reading();

    // Pins the given thread to the specified core
    void pin_thread(std::thread &t, const unsigned core) const;

    // Runs the profiler thread, keeping it alive by wrapping the functionality
    // in an infinite loop 
    void run_profiler();

};
