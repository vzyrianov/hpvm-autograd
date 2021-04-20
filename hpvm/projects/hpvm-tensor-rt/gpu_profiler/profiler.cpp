#include <cassert>
#include "profiler.h" 

Profiler::Profiler(const Device dev) : prof_dev(dev), should_run_profiler_(false), should_stop_profiler_(false) {
    // Open all streams. Not done in start_profiler() function bc the streams
    // should be strictly opened once 
    cpu_stream_.open(cpu_power_rail, std::ifstream::in);
    gpu_stream_.open(gpu_power_rail, std::ifstream::in);
    ddr_stream_.open(ddr_power_rail, std::ifstream::in);
    soc_stream_.open(soc_power_rail, std::ifstream::in);
    sys_stream_.open(sys_power_rail, std::ifstream::in);

    // Check if the jetson file id file exists to indirectly check architecture 
    std::ifstream jetson_file(jetson_chip_id);
    on_jetson_ = jetson_file.good();
    if (on_jetson_ && 
                (!cpu_stream_.is_open() || !gpu_stream_.is_open() 
                || !ddr_stream_.is_open() || !soc_stream_.is_open() 
                || !sys_stream_.is_open())) {
        std::cout << "Failed to open one of the power rails for reading\n";
        exit(1);
    }
}

Profiler::~Profiler() {
    cpu_stream_.close();
    gpu_stream_.close();
    ddr_stream_.close();
    soc_stream_.close();
    sys_stream_.close();
}

// Reinitializes boolean vars used for control flow and launches the profiler 
// thread. DOES NOT reset other internal data structures. 
void Profiler::start_profiler(){
    // Reinitialize in case the profiler object has been used before 
    should_run_profiler_ = false;
    should_stop_profiler_ = false;
    profiler_thread_ = std::thread(&Profiler::run_profiler, this);
    pin_thread(profiler_thread_, core1);
}

// Resumes the profiling of whatever executable's currently running
// DOES NOT reset any data 
void Profiler::resume_profiler() {
    std::unique_lock<std::mutex> mutex_lock(mutex_);
    if (should_run_profiler_){
       std::cout << "WARNING: resume_profiler was already called\n"; 
     }
    //std::cout<<"RESUME RESUME RESUME RESUME\n";
    should_run_profiler_ = true;
    start_time_ = clock_type::now();
    cond_var_.notify_one();
}

// Stops profiler by putting profiler thread to sleep 
void Profiler::pause_profiler() {
    std::unique_lock<std::mutex> mutex_lock(mutex_);
    if (!should_run_profiler_){
        std::cout << "WARNING: pause_profiler was already called\n";
    }
    //std::cout<<"PAUSE PAUSE PAUSE PAUSE\n";
    should_run_profiler_ = false;
    stop_time_ = clock_type::now();
    cond_var_.notify_one();
}

// Gets the delta time and total CPU/GPU and DDR energy between the last two
// calls to resume_profiler and pause_profiler
//
// Returns this as a pair of <delta time in milliseconds, energy>
std::pair<double, double> Profiler::get_time_energy() const {
    // We support taking CPU/GPU readings only
    //if(dev != CPU && dev != GPU)
      //  return std::make_pair(-1, -1);

    std::unique_lock<std::mutex> mutex_lock(vector_mutex_); // MUST use a mutex
    double total_energy = 0.0;
    if (on_jetson_) {
        //std::cout<<"power readings size"<<power_readings_.size()<<'\n';
        auto prev_time = start_time_;
        for (size_t i = 0; i < power_readings_.size(); i++){
			const auto& reading = power_readings_[i];
            std::chrono::duration<double> duration_secs = reading.time_ - prev_time;
            //if(reading.dev_type_ == dev)
            total_energy += (reading.dev_ + reading.ddr_);// * duration_secs.count();
            prev_time = reading.time_; 
        }
    }
    std::chrono::duration<double, std::milli> duration_milli = stop_time_ - start_time_;
    double delta_time = duration_milli.count();
    return std::make_pair(delta_time, total_energy);
}

void Profiler::reset() {
    std::unique_lock<std::mutex> bool_var_lock(mutex_); 
    std::unique_lock<std::mutex> vector_lock(vector_mutex_); 
    should_stop_profiler_ = false; // Can call reset after calling pause_profiler()
    should_run_profiler_ = false; // Can call reset after calling resume 
    power_readings_.clear();
    run_profiler_overhead = 0.0;
}

// Exit the profiler and kill the thread
// Must call start_profiler() to reuse this object after calling pause_profiler()
void Profiler::stop_profiler() { 
    std::cout << "Exiting profiler\n";
    should_stop_profiler_ = true;
    cond_var_.notify_one();
    profiler_thread_.join();
}

// Obtain's a single power reading from the CPU/GPU and DDR rails
void Profiler::obtain_power_reading() {
    // We support taking CPU/GPU readings only
    //if(dev != CPU && dev != GPU)
       // return;

    std::unique_lock<std::mutex> mutex_lock(vector_mutex_); // MUST use a mutex
    PowerReading reading;

    // The order matters here. All the reads have to happen together first
    // and then all the seeks have to happen together at the end, otherwise
    // there will be a significant time difference between the readings of
    // the different rails.
    reading.time_ = clock_type::now(); 
    if (on_jetson_){
        // FIXME: Use switch-case in the future.
        if(prof_dev == CPU) {
            cpu_stream_ >> reading.dev_;
            ddr_stream_ >> reading.ddr_;
            cpu_stream_.seekg(0);
        } else { 
            gpu_stream_ >> reading.dev_;
            ddr_stream_ >> reading.ddr_;
            gpu_stream_.seekg(0);
        }
	    ddr_stream_.seekg(0);
    } else {
        reading.dev_ = 0.0;
        reading.ddr_ = 0.0;
    }
    //reading.dev_type_ = dev;
    power_readings_.push_back(reading);
}

// Pins the given thread to the specified core
void Profiler::pin_thread(std::thread &t, const unsigned core) const {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    if (pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset) != 0)
        std::cout << "Couldn't set thread affinity\n";
}

// Runs the profiler thread, keeping it alive by wrapping the functionality
// in an infinite loop 
void Profiler::run_profiler(){
    while (true){
        if (should_stop_profiler_) {
            break;
        }
        // Need to lock the mutex and check the condition var 
        {
            std::unique_lock<std::mutex> mutex_lock(mutex_);
            if (should_stop_profiler_) {
                break;
            }
            // Wake the thread up when it's time to run the profiler or exit
            // the profiler 
            cond_var_.wait(mutex_lock, [this]{return should_run_profiler_
                        || should_stop_profiler_; });
        }
        if (should_stop_profiler_) {
            break;
        }
        obtain_power_reading();
    }
}

/*
// TESTS
void resume_pause_profiler(Profiler& profile_wrapper, unsigned long sleep_millis){
    profile_wrapper.resume_profiler(); 
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_millis));
    profile_wrapper.pause_profiler();

    auto time_energy_pair = profile_wrapper.get_time_energy();
    profile_wrapper.reset();
    //if (time_energy_pair.first > sleep_millis + 1 || time_energy_pair.first < sleep_millis - 1){
        printf("WARNING: time: %f, energy: %f\n", time_energy_pair.first, time_energy_pair.second);
    //}
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_millis));
    //std::cout<<"\n\n";
}

int main(){
    Profiler profile_wrapper;
    profile_wrapper.start_profiler();

    unsigned long sleep_millis = 25;
    for (size_t i = 0; i < 50; i++){
        resume_pause_profiler(profile_wrapper, sleep_millis);
    }
    // IMPORTANT
    profile_wrapper.stop_profiler();
    return 0;
}
*/
