//===----------------------------- profling.cc  ---------------------------===//
//
//===----------------------------------------------------------------------===//
//
//  This file contains code provides the definition of the interface for
// applications to start and stop profiling for energy and performance.
//
//===----------------------------------------------------------------------===//

#ifndef PROFILING_HEADER
#define PROFILING_HEADER

#include <chrono>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <stdarg.h>
#include <stdio.h>
#include <string>
#include <unordered_map>

#include "debug.h"
#include "global_data.h"

/***** Profiling routines ***/

std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
// previous_time maintains time for the latest timed operation
std::chrono::time_point<std::chrono::high_resolution_clock> previous_time;

extern "C" {

void startProfiling() {
  start_time = std::chrono::high_resolution_clock::now();
}

void stopProfiling() {
  FILE *fp = fopen("profile_data.txt", "w+");
  if (fp != NULL) {
    fwrite(profile_data.c_str(), 1, profile_data.length(), fp);
    fclose(fp);
  }

  profile_data = "";
  func_counters.clear();
}

void profileEvent(const char *event_name, bool compare_previous = false) {

  checkCudaErrors(cudaDeviceSynchronize());

  auto it = func_counters.find(event_name);
  if (it == func_counters.end()) {
    func_counters[event_name] = 1;
  } else {
    int counter = func_counters[event_name];
    counter++;
    func_counters[event_name] = counter;
  }

  std::stringstream ss;
  ss << func_counters[event_name];
  std::string event_count = ss.str();

  std::chrono::time_point<std::chrono::high_resolution_clock> zero_time;
  std::chrono::time_point<std::chrono::high_resolution_clock> time_reading =
      std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::ratio<1>> current_time =
      time_reading - zero_time;

  DEBUG("AbsoluteTime, Event = %s, Time = %f \n", event_name,
        current_time.count());
  profile_data.append(event_name);
  profile_data.append(event_count);
  profile_data.append("\t");
  profile_data.append(std::to_string(current_time.count()));

  if (compare_previous) {
    std::chrono::duration<double, std::ratio<1>> duration_time =
        time_reading - previous_time;

    profile_data.append("\t");
    profile_data.append(std::to_string(duration_time.count()));
    DEBUG("TimeDuration, Event = %s, Time = %f \n", event_name,
          duration_time.count());
  }

  profile_data.append("\n");

  previous_time = time_reading; // set the previous time reading to the current
                                // profiled time
}
}

#endif
