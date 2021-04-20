#ifndef LLVM_HPVM_RT_CONTROLLER_H
#define LLVM_HPVM_RT_CONTROLLER_H

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <vector>

#include "configuration.h"

#include "profiler.h"
#include "promise_timing_model.h"
#include <ctime>

#include <sys/stat.h>

#define ACTIVE_PROFILING
//#define JETSON_EXECUTION

/*
 * Check if a file exists
 * Return true if the file exists, false else
 */
bool fileExists(const std::string &file);

class FrequencyIndexList {
private:
  std::vector<int> idx_list;
  unsigned rep_factor;

  unsigned count;
  unsigned idx;

public:
  FrequencyIndexList(std::vector<int>, unsigned);
  unsigned getNextIndex();
};

class ProfileInfo {
private:
  // Members
  double time_total;   // Total execution time of application
  double time_compute; // Compute
  double time_control; // Control
  double time_config;  // Apply configuration

  double energy_total;   // Total energy consumed by applcation
  double energy_compute; // Compute
  double energy_control; // Control
  double energy_config;  // Apply configuration

  // Execution time of one loop iteration
  double time_compute_current_iteration; // Compute
  double time_control_current_iteration; // Control
  double time_config_current_iteration;  // Apply configuration

  // Energy comsumed by one loop iteration
  double energy_compute_current_iteration; // Compute
  double energy_control_current_iteration; // Control
  double energy_config_current_iteration;  // Apply configuration

  // Frequency of one loop iteration
  unsigned long frequency_current_iteration;

  // Vectors, where compute time and energy information
  // - for each loop iteration (outer vector)
  // - per operation (inner vector)
  //                 (tensor operation for GPU, or whole layer for PROMISE)
  // is stored
  std::vector<std::vector<std::pair<std::string, double>>> tensor_time_info;
  std::vector<std::vector<std::pair<std::string, double>>> tensor_energy_info;

  // Vectors, where total compute time and energy information per iteration are
  // stored
  std::vector<double> compute_time_info;
  std::vector<double> compute_energy_info;

  // Vectors, where control time and energy information per iteration are stored
  std::vector<double> control_time_info;
  std::vector<double> control_energy_info;

  // Vectors, where control time and energy information per iteration are stored
  std::vector<double> config_time_info;
  std::vector<double> config_energy_info;

  // Vector, where frequency information at the end of each iteration is stored
  std::vector<unsigned long> frequency_info;

  bool in_iteration;

  // Set to the path of the file where results will be written by printToFile.
  std::string out_file_name;

  // Functions
  void resetCurrentIterationTime();

  void resetCurrentIterationEnergy();

  void start_iteration();

public:
  void end_iteration();

  void addToCurrentIterationComputeTime(const char *s, double t);

  void addToCurrentIterationControlTime(double t);

  void addToCurrentIterationConfigTime(double t);

  void addToCurrentIterationComputeEnergy(const char *s, double e);

  void addToCurrentIterationControlEnergy(double e);

  void addToCurrentIterationConfigEnergy(double e);

  double getTotalTime();

  double getTotalEnergy();

  double getCurrentIterationComputeTime();

  double getCurrentIterationComputeEnergy();

  void readIterationFrequency();

  unsigned long getIterationFrequency();

  void set_out_file_name(const std::string &str);

  void printToFile();

  ProfileInfo();
};

class Slowdowns {
private:
  std::vector<float> slowdowns;
  unsigned idx;

public:
  Slowdowns();

  unsigned getSlowdownsNumber();

  float getNextSlowdown();
};

class RuntimeController;

extern RuntimeController *RC;

class RuntimeController {
private:
  // Members

  // Configurations.
  // Configurations initially read - all generated from autotuner
  std::vector<struct Configuration> InitialConfigurations;

  // The ones in non dominated set (of pareto optimal points)
  // for accuracy loss-speedup
  std::vector<struct Configuration *> SpeedupConfigurations;
  // The ones in non dominated set (of pareto optimal points)
  // for accuracy loss-energy
  std::vector<struct Configuration *> EnergyConfigurations;
  // The ones in non dominated set (of pareto optimal points)
  // for accuracy loss-speedup-energy
  std::vector<struct Configuration *> ThreeDCurveConfigurations;

  std::vector<struct Configuration *> *Configurations;
  unsigned configurationIdx = 0;

  double baseline_time = 0.0; // Execution time of baseline configuration
  Slowdowns *slowdowns;

  float pseudo_rd = 0.0;
  std::mt19937 generator;
  std::uniform_real_distribution<> distr;

  // Stored frequency and speedup for a running iteration.
  // To be changed for a new frequency point
  unsigned long g_freq = 0;
  double g_speedup = 1.0;

  /*** Objects used to gather timing and energy information for execution ***/
  ProfileInfo *PI;
  Profiler *profiler;
  Promise *promise;

  // Frequency Index List: used to provide a list of indices, that are used to
  // update the frequency of the Jetson board
  FrequencyIndexList *FIL;

  // Functions

  // Private functions of profiler
  void start_profiler();
  void stop_profiler();

  void setProfileInfoFilename(const char *);
  void readConfigurationFile(const char *);

  void computeParetoConfigurationPoints();
  void compute3DParetoConfigurationPoints();

public:
  // For testing purposes only - do not use widely
  std::vector<struct Configuration *> &getSpeedupConfigurations();
  // For testing purposes only - do not use widely
  std::vector<struct Configuration *> &getEnergyConfigurations();
  // For testing purposes only - do not use widely
  std::vector<struct Configuration *> &getThreeDCurveConfigurations();
  // For testing purposes only - do not use widely
  unsigned getConfigurationIdx();

  double getCurrentConfigurationSpeedup();
  double getCurrentConfigurationEnergy();
  double getCurrentConfigurationAccuracy();
  double getCurrentConfigurationAccuracyLoss();

  NodeConfiguration *getNodeConfiguration(const char *data);

  // Functions for runtime control
  unsigned long getLastFrequency();
  void setLastFrequency(unsigned long);
  double getLastSpeedup();
  void setLastSpeedup(double);

  void findNextConfiguration();
  void findTargetConfiguration(float, enum SEARCH_KIND);
  void adjustTargetConfiguration(float);
  float getGoalSpeedup();
  double getBaselineTime();
  Slowdowns *getSlowdowns();

  void init(const char *Cstr);

  // Exposing functionality of ProfileInfo
  void end_iteration();

  void addToCurrentIterationComputeTime(const char *s, double t);

  void addToCurrentIterationControlTime(double t);

  void addToCurrentIterationConfigTime(double t);

  void addToCurrentIterationComputeEnergy(const char *s, double e);

  void addToCurrentIterationControlEnergy(double e);

  void addToCurrentIterationConfigEnergy(double e);

  double getCurrentIterationComputeTime();

  double getCurrentIterationComputeEnergy();

  void readIterationFrequency();

  unsigned long getIterationFrequency();

  void updateFrequency();

  void writeProfileInfo();

  // Exposing functionality of (gpu) profiler
  void resume_profiler();

  void pause_profiler();

  void reset_profiler();

  std::pair<double, double> get_time_energy() const;

  // Exposing functionality of promise simulator
  std::pair<double, double>
  fc_profile(const unsigned num_rows_a, const unsigned num_cols_a,
             const unsigned num_rows_b, const unsigned num_cols_b,
             const unsigned voltage_swing, const unsigned patch_factor);

  std::pair<double, double>
  conv_profile(const unsigned n, const unsigned c, const unsigned h,
               const unsigned w, const unsigned c_out, const unsigned c_in,
               const unsigned k_h, const unsigned k_w, const unsigned s_h,
               const unsigned s_w, const unsigned voltage_swing,
               const unsigned patch_factor);

  // Constructor and descructor
  RuntimeController();

  ~RuntimeController();

  // Helper Functions
  void printConfigurations(std::vector<struct Configuration> &);
  void printConfigurations(std::vector<struct Configuration *> &);
};
#define NODE_NAME_BUFFER_SIZE 10
#define AL_THRESHOLD 0.01
#define MAX_GOAL_SPEEDUP 9

//*** Methods to compute accuracy of a tensor by the runtime controller   ***//

uint32_t *hpvm_rt_readLabelsBatch_cached(const char *labels_file, int start,
                                         int end);

//*** Copied from dnn_sources/include/utils.h                             ***//
float hpvm_rt_computeAccuracy3(uint32_t *labels, void *result_ptr);

#endif
