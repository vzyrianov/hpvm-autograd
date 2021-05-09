//===--------------------------- hpvm-rt-controller.cpp
//---------------------===//
//
//===----------------------------------------------------------------------===//
//
//  This file contains code for HPVM Dynamic Approximation Control.
//
//  The runtime controller:
//    * Reads in the configuration file passed to the HPVM binary
//    * Contructs a Pareto Curve
//    * Based on the selected Mode it switches configurations at runtime
//
//   Author: Maria Kotsifakou
//
//===----------------------------------------------------------------------===//


// ***NOTE*** The macro definitions below control the runtime policy

//--- llvm_hpvm_invokeRtControl_BASE is the baseline policy (default) that just uses the first config (configuration file)
#define llvm_hpvm_invokeRtControl_BASE llvm_hpvm_invokeRtControl
//--- llvm_hpvm_invokeRtControl_ADJUST_PR is the probabilistic config selection from Pareto curve - Uncomment to use
//#define llvm_hpvm_invokeRtControl_ADJUST_PR llvm_hpvm_invokeRtControl



#include "hpvm-rt-controller.h"
#include "global_data.h"
#include "jetson_freq_utils.h"
#include <fstream>


//---------------------------------------------------------------------------//


// Functions
void ProfileInfo::resetCurrentIterationTime() {
  time_compute_current_iteration = 0.0;
  time_control_current_iteration = 0.0;
  time_config_current_iteration = 0.0;
}

void ProfileInfo::resetCurrentIterationEnergy() {
  energy_compute_current_iteration = 0.0;
  energy_control_current_iteration = 0.0;
  energy_config_current_iteration = 0.0;
}

void ProfileInfo::start_iteration() {
  if (!in_iteration) {
    resetCurrentIterationTime();
    resetCurrentIterationEnergy();
    tensor_time_info.push_back(std::vector<std::pair<std::string, double>>());
    tensor_energy_info.push_back(std::vector<std::pair<std::string, double>>());
    in_iteration = true;
  }
}
void ProfileInfo::end_iteration() {
  // Update time counters
  time_compute += time_compute_current_iteration;
  time_control += time_control_current_iteration;
  time_config += time_config_current_iteration;

  time_total +=
      (time_compute_current_iteration + time_control_current_iteration +
       time_config_current_iteration);

  // Update energy counters
  energy_compute += energy_compute_current_iteration;
  energy_control += energy_control_current_iteration;
  energy_config += energy_config_current_iteration;

  energy_total +=
      (energy_compute_current_iteration + energy_control_current_iteration +
       energy_config_current_iteration);

  // Save current iteration counters
  compute_time_info.push_back(time_compute_current_iteration);
  compute_energy_info.push_back(energy_compute_current_iteration);
  control_time_info.push_back(time_control_current_iteration);
  control_energy_info.push_back(energy_control_current_iteration);
  config_time_info.push_back(time_config_current_iteration);
  config_energy_info.push_back(energy_config_current_iteration);

  frequency_info.push_back(frequency_current_iteration);

  // Note end of iteration
  in_iteration = false;
}


void ProfileInfo::addToCurrentIterationComputeTime(const char *s, double t) {
  start_iteration();
  time_compute_current_iteration += t;
  tensor_time_info.back().push_back(std::make_pair(std::string(s), t));
}

void ProfileInfo::addToCurrentIterationControlTime(double t) {
  start_iteration();
  time_control_current_iteration += t;
}

void ProfileInfo::addToCurrentIterationConfigTime(double t) {
  start_iteration();
  time_config_current_iteration += t;
}

void ProfileInfo::addToCurrentIterationComputeEnergy(const char *s, double e) {
  start_iteration();
  energy_compute_current_iteration += e;
  tensor_energy_info.back().push_back(std::make_pair(std::string(s), e));
}

void ProfileInfo::addToCurrentIterationControlEnergy(double e) {
  start_iteration();
  energy_control_current_iteration += e;
}

void ProfileInfo::addToCurrentIterationConfigEnergy(double e) {
  start_iteration();
  energy_config_current_iteration += e;
}

double ProfileInfo::getTotalTime() { return time_total; }

double ProfileInfo::getTotalEnergy() { return energy_total; }

double ProfileInfo::getCurrentIterationComputeTime() {
  return time_compute_current_iteration;
}

double ProfileInfo::getCurrentIterationComputeEnergy() {
  return energy_compute_current_iteration;
}

void ProfileInfo::set_out_file_name(const std::string &str) {
  out_file_name = str;
}

void ProfileInfo::printToFile() {
  INFO("Writing Runtime Profile Info File...\n");

  if (control_time_info.size() == 0)
    return;

  std::ofstream s_out(out_file_name.c_str());
  if (!s_out) {
    ERROR("Failed to open output file.");
    abort();
  }

  // By construction, tensor_time_info and tensor_energy_info are expected
  // to have equal sizes, in outer and inner vectors both,
  // and all time_info and energy_info vectors must have the same size.
  unsigned iterations = tensor_time_info.size();
  CUSTOM_ASSERT((tensor_time_info.size() == iterations) &&
                (tensor_energy_info.size() == iterations) &&
                (control_time_info.size() == iterations) &&
                (control_energy_info.size() == iterations) &&
                (config_time_info.size() == iterations) &&
                (config_energy_info.size() == iterations) &&
                (frequency_info.size() == iterations) &&
                "time_info, energy_info, frequency_info size: \
                   iteration number does not match.");

  for (unsigned i = 0; i < tensor_time_info.size(); i++) {
    // time_info.size() == energy_info.size(), since we passed the assertion
    s_out << "Iteration " << i << "\n";

    CUSTOM_ASSERT(
        (tensor_time_info[i].size() == tensor_energy_info[i].size()) &&
        "time_info and energy_info size: operation number does not match.");
    for (unsigned j = 0; j < tensor_time_info[i].size(); j++) {
      // time_info[i].size() == energy_info[i].size(), we passed the assertion
      CUSTOM_ASSERT(
          (tensor_time_info[i][j].first == tensor_energy_info[i][j].first) &&
          "time_info and energy_info: operation does not match.");
      s_out << tensor_time_info[i][j].first << " "
            << tensor_time_info[i][j].second << " "
            << tensor_energy_info[i][j].second << "\n";
    }

    s_out << "\nIteration Compute Time   : " << compute_time_info[i] << "\n";
    s_out << "Iteration Compute Energy : " << compute_energy_info[i] << "\n";
    s_out << "Iteration Control Time   : " << control_time_info[i] << "\n";
    s_out << "Iteration Control Energy : " << control_energy_info[i] << "\n";
    s_out << "Iteration Config Time   : " << config_time_info[i] << "\n";
    s_out << "Iteration Config Energy : " << config_energy_info[i] << "\n";
    s_out << "Iteration End Frequency : " << frequency_info[i] << "\n\n\n";
  }
  s_out << "\n\nTotal Compute Time  : " << time_compute << "\n";
  s_out << "Total Compute Energy: " << energy_compute << "\n";

  s_out << "\nTotal Control Time  : " << time_control << "\n";
  s_out << "Total Control Energy: " << energy_control << "\n";

  s_out << "\nTotal Config Time  : " << time_config << "\n";
  s_out << "Total Config Energy: " << energy_config << "\n";

  s_out << "\nTotal Time  : " << time_total << "\n";
  s_out << "Total Energy: " << energy_total << "\n";

  s_out.close();

  INFO("Done writing profile.\n");
}

ProfileInfo::ProfileInfo()
    : time_total(0.0), energy_total(0.0), time_compute_current_iteration(0.0),
      time_control_current_iteration(0.0), time_config_current_iteration(0.0),
      energy_compute_current_iteration(0.0),
      energy_control_current_iteration(0.0),
      energy_config_current_iteration(0.0), frequency_current_iteration(0),
      in_iteration(false) {}

Slowdowns::Slowdowns() {

  idx = 0;
  std::ifstream s_in("slowdowns.txt");
  if (!s_in) {
    DEBUG("slowdowns file not found. Initializing slowdowns randomly.\n");
    for (unsigned i = 0; i < 10; i++) {
      slowdowns.push_back(1.0 + (rand() / (RAND_MAX / (5.0 - 1.0))));
    }
  } else {
    DEBUG("Found slowdowns file.\n");
    for (std::string line; std::getline(s_in, line);) {
      float s = std::stof(line);
      slowdowns.push_back(s);
    }
  }
}

unsigned Slowdowns::getSlowdownsNumber() { return slowdowns.size(); }

float Slowdowns::getNextSlowdown() {
  float tmp = slowdowns[idx];
  idx = (idx + 1) % slowdowns.size();
  return tmp;
}

RuntimeController *RC;

// Functions

// Private functions of profiler
void RuntimeController::start_profiler() {
  if (profiler)
    profiler->start_profiler();
}
void RuntimeController::stop_profiler() {
  if (profiler)
    profiler->stop_profiler();
}
// For testing purposes only - do not use widely
std::vector<struct Configuration *> &
RuntimeController::getSpeedupConfigurations() {
  return SpeedupConfigurations;
}
// For testing purposes only - do not use widely
std::vector<struct Configuration *> &
RuntimeController::getEnergyConfigurations() {
  return EnergyConfigurations;
}
// For testing purposes only - do not use widely
std::vector<struct Configuration *> &
RuntimeController::getThreeDCurveConfigurations() {
  return ThreeDCurveConfigurations;
}
// For testing purposes only - do not use widely
unsigned RuntimeController::getConfigurationIdx() { return configurationIdx; }

double RuntimeController::getCurrentConfigurationSpeedup() {
  return (double)(*Configurations)[configurationIdx]->speedup;
}

double RuntimeController::getCurrentConfigurationEnergy() {
  return (double)(*Configurations)[configurationIdx]->energy;
}

double RuntimeController::getCurrentConfigurationAccuracy() {
  return (double)(*Configurations)[configurationIdx]->accuracy;
}

double RuntimeController::getCurrentConfigurationAccuracyLoss() {
  return (double)(*Configurations)[configurationIdx]->accuracyLoss;
}

NodeConfiguration *RuntimeController::getNodeConfiguration(const char *data) {

  // if visc.node.id Not specified for this HPVM Node
  if (currentTensorID == ~0U) {
    std::string s(data);
    // All nodes are expected to have a configuration
    return (*Configurations)[configurationIdx]->setup.at(s);
  } else {
    DEBUG("-- currentTensorID = %u \n", currentTensorID);
    return (*Configurations)[configurationIdx]->idConfigMap.at(currentTensorID);
  }
}

void RuntimeController::init(const char *Cstr) {
  INFO("INIT RUNTIME CONTROLLER ==================\n");
  printf("INIT RUNTIME CONTROLLER ==================\n");
  // We initialize the path to the profile info output file,
  // based on the path given for the configuration file
  PI->set_out_file_name("profile_info.txt");
  readConfigurationFile(Cstr);

  // NOTE: Configurations is pareto-configs. InitialConfigurations is the full
  // list (config file)
  Configurations = NULL;
  computeParetoConfigurationPoints();
  //    compute3DParetoConfigurationPoints(); Not using 3D curve
  INFO("Speedup Configurations\n");
  printConfigurations(SpeedupConfigurations);

  configurationIdx = 0; 
  Configurations = &SpeedupConfigurations;

  // Initializations for different runtime control strategies
  srand(static_cast<unsigned>(time(0)));
  slowdowns = new Slowdowns();

  // Pseudo random variable (when we did few experiments)
  // or true random numbers for probabilistic control
  pseudo_rd = 0.0;
  std::random_device rd; // Will be used to obtain a seed for the random number engine
  generator = std::mt19937(rd()); // Standard mersenne_twister_engine seeded with rd()
  distr = std::uniform_real_distribution<>(0.0, 1.0);

  g_freq = available_freqs[13];
  g_speedup = 1.0;

  // Initialize utility objects for knob reading
  perfParamSet = new PerfParamSet();
  sampParamSet = new SampParamSet();

  // Start profiling thread in the background, ready to time
  start_profiler();
  pause_profiler();
  reset_profiler();
}

// Exposing functionality of ProfileInfo
void RuntimeController::end_iteration() {
  if (PI)
    PI->end_iteration();
}

void RuntimeController::addToCurrentIterationComputeTime(const char *s,
                                                         double t) {
  if (PI)
    PI->addToCurrentIterationComputeTime(s, t);
}

void RuntimeController::addToCurrentIterationControlTime(double t) {
  if (PI)
    PI->addToCurrentIterationControlTime(t);
}

void RuntimeController::addToCurrentIterationConfigTime(double t) {
  if (PI)
    PI->addToCurrentIterationConfigTime(t);
}

void RuntimeController::addToCurrentIterationComputeEnergy(const char *s,
                                                           double e) {
  if (PI)
    PI->addToCurrentIterationComputeEnergy(s, e);
}

void RuntimeController::addToCurrentIterationControlEnergy(double e) {
  if (PI)
    PI->addToCurrentIterationControlEnergy(e);
}

void RuntimeController::addToCurrentIterationConfigEnergy(double e) {
  if (PI)
    PI->addToCurrentIterationConfigEnergy(e);
}

double RuntimeController::getCurrentIterationComputeTime() {
  return (PI ? PI->getCurrentIterationComputeTime() : 0.0);
}

double RuntimeController::getCurrentIterationComputeEnergy() {
  return (PI ? PI->getCurrentIterationComputeEnergy() : 0.0);
}


void RuntimeController::writeProfileInfo() {
  if (PI)
    PI->printToFile();
}

// Exposing functionality of (gpu) profiler
void RuntimeController::resume_profiler() {
  if (profiler)
    profiler->resume_profiler();
}

void RuntimeController::pause_profiler() {
  if (profiler)
    profiler->pause_profiler();
}

void RuntimeController::reset_profiler() {
  if (profiler)
    profiler->reset();
}

std::pair<double, double> RuntimeController::get_time_energy() const {
  return (profiler ? profiler->get_time_energy() : std::make_pair(0.0, 0.0));
}

// Exposing functionality of promise simulator
std::pair<double, double> RuntimeController::fc_profile(
    const unsigned num_rows_a, const unsigned num_cols_a,
    const unsigned num_rows_b, const unsigned num_cols_b,
    const unsigned voltage_swing, const unsigned patch_factor) {

  return (promise ? promise->fc_profile(num_rows_a, num_cols_a, num_rows_b,
                                        num_cols_b, voltage_swing, patch_factor)
                  : std::make_pair(0.0, 0.0));
}

std::pair<double, double> RuntimeController::conv_profile(
    const unsigned n, const unsigned c, const unsigned h, const unsigned w,
    const unsigned c_out, const unsigned c_in, const unsigned k_h,
    const unsigned k_w, const unsigned s_h, const unsigned s_w,
    const unsigned voltage_swing, const unsigned patch_factor) {
  
  return (promise ? promise->conv_profile(n, c, h, w, c_out, c_in, k_h, k_w,
                                          s_h, s_w, voltage_swing, patch_factor)
                  : std::make_pair(0.0, 0.0));
}

// Constructor and descructor
RuntimeController::RuntimeController() {
  configurationIdx = 0;

  // NOTE: The 14 Frequency levels are specific to NVIDIA Jetson Tx2
  // More Frequency utils (not used by default) present in include/jetson_freq_utils.h 
  FIL = new FrequencyIndexList({13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0},
                               10);
  
#ifdef ACTIVE_PROFILING
  PI = new ProfileInfo();
  profiler = new Profiler();
  promise = new Promise();
#else
  PI = NULL;
  profiler = NULL;
  promise = NULL;
#endif
}

RuntimeController::~RuntimeController() {

  stop_profiler();
  writeProfileInfo();

  if (PI) {
    delete PI;
  }
  if (profiler) {
    delete profiler;
  }
  if (promise) {
    delete promise;
  }

  for (std::vector<struct Configuration>::iterator
           it = InitialConfigurations.begin(),
           ie = InitialConfigurations.end();
       it != ie; ++it) {
    std::map<std::string, NodeConfiguration *> ConfSetup = it->setup;
    for (std::map<std::string, NodeConfiguration *>::const_iterator it =
             ConfSetup.begin();
         it != ConfSetup.end(); ++it) {
      delete it->second;
    }
  }
  // Handle freeing memory, for all configurations
  // A way to do that is to not free the initial configurations in the pareto
  // curve, and free all at once in the end This is done because configurations
  // are stored in different containers, but share the node setup
}

void RuntimeController::readConfigurationFile(const char *str) {

  INFO("Reading Configuration File...\n");

  std::ifstream qin(str);

  if (!qin) {
    ERROR("Failed to open configuration file.");
    abort();
  }

  bool readingFirstLine = false;

  // Read baseline_time from first line of configuration file
  std::string first_line;
  std::getline(qin, first_line);
  DEBUG("first_line: %s\n", first_line.c_str());

  try {
    baseline_time = std::stod(first_line);
    DEBUG("Baseline time: %lf\n\n", baseline_time);
  } catch (...) {
    ERROR("Please Add/Fix Baseline Time at Top of Config File.. ");
  }

  unsigned int firstTensorID = 1;
  for (std::string line; std::getline(qin, line);) {
    DEBUG("line: %s\n", line.c_str());

    // Tokenize using ' ' as delimiter
    // Vector to store tokens
    std::vector<std::string> tokens;

    for (auto i = strtok(&line[0], " "); i != NULL; i = strtok(NULL, " "))
      tokens.push_back(i);

    for (unsigned i = 0; i < tokens.size(); i++)
      DEBUG("t: %s\n", tokens[i].c_str());

    DEBUG("\n");

    if (tokens[0] == "+++++") { // Found new configuration start token
      // Mark the start of a new configuration
      readingFirstLine = true;
      continue;
    }

    if (tokens[0] == "-----") { // Found configuration end token
      // Mark the end of current configuration
      continue;
    }

    if (readingFirstLine) {
      // Read first line, to create the new configuration struct
      readingFirstLine = false;
      firstTensorID = 1; // reset first tensor ID for new config

      InitialConfigurations.push_back(
          Configuration(tokens[0], std::stof(tokens[1]), std::stof(tokens[2]),
                        std::stof(tokens[3]), std::stof(tokens[4])));
      continue;
    }

    if (tokens[1] == "gpu") {
      DEBUG("Found gpu configuration\n");

      // There must be at least one operation, with an approximation option
      CUSTOM_ASSERT((tokens.size() >= 5) &&
                    "Not enough operations - approximation options.");

      GPUNodeConfiguration *NodeConf = new GPUNodeConfiguration();
      InitialConfigurations.back().setup.insert(
          std::make_pair(tokens[0], NodeConf));

      // Updating map of visc.node.id ID values to NodeConfigurations
      // FIXME: Do same for CPU and PROMISE configs
      InitialConfigurations.back().idConfigMap.insert(
          std::make_pair(firstTensorID, NodeConf));
      DEBUG("*** firstTensorID = %d \n\n", firstTensorID);

      unsigned idx = 2;
      while (idx < tokens.size()) {
        if (tokens[idx] == "add") {
          DEBUG("Found add operation\n");
          NodeConf->pushNewTensorOperation(
              GPUNodeConfiguration::TENSOR_OP::ADD);
          idx++;
        } else if (tokens[idx] == "batchnorm") {
          DEBUG("Found batchnorm operation\n");
          NodeConf->pushNewTensorOperation(
              GPUNodeConfiguration::TENSOR_OP::BATCHNORM);
          idx++;
        } else if (tokens[idx] == "conv") {
          DEBUG("Found conv operation\n");
          NodeConf->pushNewTensorOperation(
              GPUNodeConfiguration::TENSOR_OP::CONV);
          idx++;
        } else if (tokens[idx] == "group_conv") {
          DEBUG("Found group_conv operation\n");
          NodeConf->pushNewTensorOperation(
              GPUNodeConfiguration::TENSOR_OP::GROUP_CONV);
          idx++;
        } else if (tokens[idx] == "mul") {
          DEBUG("Found mul operation\n");
          NodeConf->pushNewTensorOperation(
              GPUNodeConfiguration::TENSOR_OP::MUL);
          idx++;
        } else if (tokens[idx] == "relu") {
          DEBUG("Found relu operation\n");
          NodeConf->pushNewTensorOperation(
              GPUNodeConfiguration::TENSOR_OP::RELU);
          idx++;
        } else if (tokens[idx] == "clipped_relu") {
          DEBUG("Found clipped_relu operation\n");
          NodeConf->pushNewTensorOperation(
              GPUNodeConfiguration::TENSOR_OP::CLIPPED_RELU);
          idx++;
        } else if (tokens[idx] == "tanh") {
          DEBUG("Found tanh operation\n");
          NodeConf->pushNewTensorOperation(
              GPUNodeConfiguration::TENSOR_OP::TANH);
          idx++;
        } else if (tokens[idx] == "pool_max") {
          DEBUG("Found pool_max operation\n");
          NodeConf->pushNewTensorOperation(
              GPUNodeConfiguration::TENSOR_OP::POOL_MAX);
          idx++;
        } else if (tokens[idx] == "pool_mean") {
          DEBUG("Found pool_mean operation\n");
          NodeConf->pushNewTensorOperation(
              GPUNodeConfiguration::TENSOR_OP::POOL_MEAN);
          idx++;
        } else if (tokens[idx] == "pool_min") {
          DEBUG("Found pool_min operation\n");
          NodeConf->pushNewTensorOperation(
              GPUNodeConfiguration::TENSOR_OP::POOL_MIN);
          idx++;
        } else if (tokens[idx] == "softmax") {
          DEBUG("Found softmax operation\n");
          NodeConf->pushNewTensorOperation(
              GPUNodeConfiguration::TENSOR_OP::SOFTMAX);
          idx++;
        } else /*Not a new operation. This means an approximation option*/
            if (tokens[idx] == "fp32") {
          DEBUG("Found fp32 option\n");
          int fp32 = std::stoi(tokens[idx + 1]);
          DEBUG("fp32 parameter: %d, ignoring\n", fp32);
          NodeConf->pushNewApproximationChoiceForOperation(
              GPUNodeConfiguration::APPROX::FP32, fp32);
          idx += 2;
        } else if (tokens[idx] == "fp16") {
          DEBUG("Found fp16 option\n");
          int fp16 = std::stoi(tokens[idx + 1]);
          DEBUG("fp16 parameter: %d, ignoring\n", fp16);
          NodeConf->pushNewApproximationChoiceForOperation(
              GPUNodeConfiguration::APPROX::FP16, fp16);
          idx += 2;
        } else if (tokens[idx] == "perf") {
          DEBUG("Found perf option\n");
          int perf = std::stoi(tokens[idx + 1]);
          DEBUG("perf parameter: %d\n", perf);
          NodeConf->pushNewApproximationChoiceForOperation(
              GPUNodeConfiguration::APPROX::PERFORATION, perf);
          idx += 2;
        } else if (tokens[idx] == "perf_fp16") {
          DEBUG("Found perf_fp16 option\n");
          int perf_fp16 = std::stoi(tokens[idx + 1]);
          DEBUG("perf_fp16 parameter: %d\n", perf_fp16);
          NodeConf->pushNewApproximationChoiceForOperation(
              GPUNodeConfiguration::APPROX::PERFORATION_HP, perf_fp16);
          idx += 2;
        } else if (tokens[idx] == "samp") {
          DEBUG("Found samp option\n");
          int samp = std::stoi(tokens[idx + 1]);
          DEBUG("samp parameter: %d\n", samp);
          NodeConf->pushNewApproximationChoiceForOperation(
              GPUNodeConfiguration::APPROX::INPUT_SAMPLING, samp);
          idx += 2;
        } else if (tokens[idx] == "samp_fp16") {
          DEBUG("Found samp_fp16 option\n");
          int samp_fp16 = std::stoi(tokens[idx + 1]);
          DEBUG("samp_fp16 parameter: %d\n", samp_fp16);
          NodeConf->pushNewApproximationChoiceForOperation(
              GPUNodeConfiguration::APPROX::INPUT_SAMPLING_HP, samp_fp16);
          idx += 2;
        } else if (tokens[idx] == "red_samp") {
          DEBUG("Found red_samp option\n");
          int red_samp = std::stoi(tokens[idx + 1]);
          DEBUG("red_samp parameter: %d\n", red_samp);
          NodeConf->pushNewApproximationChoiceForOperation(
              GPUNodeConfiguration::APPROX::REDUCTION_SAMPLING, red_samp);
          idx += 2;
        }
        // TODO: other approximation options handled here
      }

      // Update first TensorID using number of tensor ops in current node
      firstTensorID += NodeConf->getApproxChoices().size();

    } else if (tokens[1] == "cpu") {

      // There must be at least one operation, with an approximation option
      CUSTOM_ASSERT((tokens.size() >= 5) &&
                    "Not enough operations - approximation options.");

      CPUNodeConfiguration *NodeConf = new CPUNodeConfiguration();
      InitialConfigurations.back().setup.insert(
          std::make_pair(tokens[0], NodeConf));

      InitialConfigurations.back().idConfigMap.insert(
          std::make_pair(firstTensorID, NodeConf));

      unsigned idx = 2;
      while (idx < tokens.size()) {
        if (tokens[idx] == "add") {
          DEBUG("Found add operation\n");
          NodeConf->pushNewTensorOperation(
              CPUNodeConfiguration::TENSOR_OP::ADD);
          idx++;
        } else if (tokens[idx] == "batchnorm") {
          DEBUG("Found batchnorm operation\n");
          NodeConf->pushNewTensorOperation(
              CPUNodeConfiguration::TENSOR_OP::BATCHNORM);
          idx++;
        } else if (tokens[idx] == "conv") {
          DEBUG("Found conv operation\n");
          NodeConf->pushNewTensorOperation(
              CPUNodeConfiguration::TENSOR_OP::CONV);
          idx++;
        } else if (tokens[idx] == "group_conv") {
          DEBUG("Found group_conv operation\n");
          NodeConf->pushNewTensorOperation(
              CPUNodeConfiguration::TENSOR_OP::GROUP_CONV);
          idx++;
        } else if (tokens[idx] == "mul") {
          DEBUG("Found mul operation\n");
          NodeConf->pushNewTensorOperation(
              CPUNodeConfiguration::TENSOR_OP::MUL);
          idx++;
        } else if (tokens[idx] == "relu") {
          DEBUG("Found relu operation\n");
          NodeConf->pushNewTensorOperation(
              CPUNodeConfiguration::TENSOR_OP::RELU);
          idx++;
        } else if (tokens[idx] == "clipped_relu") {
          DEBUG("Found clipped_relu operation\n");
          NodeConf->pushNewTensorOperation(
              CPUNodeConfiguration::TENSOR_OP::CLIPPED_RELU);
          idx++;
        } else if (tokens[idx] == "tanh") {
          DEBUG("Found tanh operation\n");
          NodeConf->pushNewTensorOperation(
              CPUNodeConfiguration::TENSOR_OP::TANH);
          idx++;
        } else if (tokens[idx] == "pool_max") {
          DEBUG("Found pool_max operation\n");
          NodeConf->pushNewTensorOperation(
              CPUNodeConfiguration::TENSOR_OP::POOL_MAX);
          idx++;
        } else if (tokens[idx] == "pool_mean") {
          DEBUG("Found pool_mean operation\n");
          NodeConf->pushNewTensorOperation(
              CPUNodeConfiguration::TENSOR_OP::POOL_MEAN);
          idx++;
        } else if (tokens[idx] == "pool_min") {
          DEBUG("Found pool_min operation\n");
          NodeConf->pushNewTensorOperation(
              CPUNodeConfiguration::TENSOR_OP::POOL_MIN);
          idx++;
        } else if (tokens[idx] == "softmax") {
          DEBUG("Found softmax operation\n");
          NodeConf->pushNewTensorOperation(
              CPUNodeConfiguration::TENSOR_OP::SOFTMAX);
          idx++;
        } else /*Not a new operation. This means an approximation option*/
            if (tokens[idx] == "fp32") {
          DEBUG("Found fp32 option\n");
          int fp32 = std::stoi(tokens[idx + 1]);
          DEBUG("fp32 parameter: %d, ignoring\n", fp32);
          NodeConf->pushNewApproximationChoiceForOperation(
              CPUNodeConfiguration::APPROX::FP32, fp32);
          idx += 2;
        } else if (tokens[idx] == "perf") {
          DEBUG("Found perf option\n");
          int perf = std::stoi(tokens[idx + 1]);
          DEBUG("perf parameter: %d\n", perf);
          NodeConf->pushNewApproximationChoiceForOperation(
              CPUNodeConfiguration::APPROX::PERFORATION, perf);
          idx += 2;
        } else if (tokens[idx] == "samp") {
          DEBUG("Found samp option\n");
          int samp = std::stoi(tokens[idx + 1]);
          DEBUG("samp parameter: %d\n", samp);
          NodeConf->pushNewApproximationChoiceForOperation(
              CPUNodeConfiguration::APPROX::INPUT_SAMPLING, samp);
          idx += 2;
        }
        // TODO: other approximation options handled here
      }
      firstTensorID += NodeConf->getApproxChoices().size();
    } else {
      DEBUG("Invalid Configuration File\n");
      exit(1);
    }
  }

  qin.close();
  DEBUG("DONE.\n");
}
void RuntimeController::computeParetoConfigurationPoints() {

  // Keep indices of pareto optimal points (configurations from
  // InitialConfigurations vector that were copied to Configurations vector.)
  // The others' setup pointer needs to be deleted
  std::vector<unsigned> Indices;

  // Baseline configuration (first one we read) always belongs to the curve
  SpeedupConfigurations.push_back(&InitialConfigurations[0]);
  EnergyConfigurations.push_back(&InitialConfigurations[0]);

  // Sort the configurations according to accuracy loss
  INFO("Sorting autotuner configurations...\n");
  std::sort(InitialConfigurations.begin() + 1, InitialConfigurations.end(),
            ConfigurationLessThan());
  INFO("Done sorting.\n");

  for (unsigned start_idx = 1; start_idx < InitialConfigurations.size();) {
    // Points to first Configuration with different (higher) accuracy loss
    // compared to the one pointed by start_idx
    unsigned end_idx = start_idx + 1;
    while ((end_idx < InitialConfigurations.size()) &&
           (InitialConfigurations[end_idx].accuracyLoss -
                InitialConfigurations[start_idx].accuracyLoss <
            AL_THRESHOLD)) {
      end_idx++;
    }
    DEBUG("start_idx = %d, end_idx = %d\n", start_idx, end_idx);
    // Now, all elements in [start_idx, end_idx) have equal accuracy loss,
    // that is lower from later ones.

    // Find the best speedup and energy between them as well
    float sp = -1.0; // FLT_MIN
    unsigned sp_idx = 0;

    float en = -1.0; // FLT_MIN
    unsigned en_idx = 0;

    for (unsigned i = start_idx; i < end_idx; i++) {
      if (InitialConfigurations[i].speedup > sp) {
        sp = InitialConfigurations[i].speedup;
        sp_idx = i;
      }
      if (InitialConfigurations[i].energy > en) {
        en = InitialConfigurations[i].energy;
        en_idx = i;
      }
    }
    DEBUG("accuracy loss = %f, speedup = %f, at sp_idx = %d\n",
          InitialConfigurations[sp_idx].accuracyLoss, sp, sp_idx);
    // Found best speedup for this accuracy point (not dominated by any of
    // these).
    DEBUG("accuracy loss = %f, energy = %f, at en_idx = %d\n",
          InitialConfigurations[en_idx].accuracyLoss, en, en_idx);
    // Found best energy for this accuracy point (not dominated by any of
    // these).

    // Now, we need to check that it is not dominated.
    // - better accuracy loss of all in initial configurations out of
    // start_idx, end_idx range
    // - better or equal speedup to the ones within this range
    // We only need to check the points already in Configurations, that have
    // already been inserted in pareto frontier. These have better accuracy
    // loss, so this one will only be added if it shows better speedup
    // The one in curve with best speedup so far is the last one (with worst
    // = highest accuracy loss), so compare only with that one.

    // Similar handling of energy vector

    bool sp_notDominated = true;
    if (!SpeedupConfigurations.empty()) {
      if (SpeedupConfigurations.back()->speedup >= sp)
        sp_notDominated = false;
    }

    bool en_notDominated = true;
    if (!EnergyConfigurations.empty()) {
      if (EnergyConfigurations.back()->energy >= en)
        en_notDominated = false;
    }

    DEBUG("sp_notDominated = %d\n", sp_notDominated);
    DEBUG("en_notDominated = %d\n", en_notDominated);

    // If not dominated, insert in pareto frontier set
    if (sp_notDominated) {
      SpeedupConfigurations.push_back(&InitialConfigurations[sp_idx]);
    }
    if (en_notDominated) {
      EnergyConfigurations.push_back(&InitialConfigurations[en_idx]);
    }

    // Keep track of unnecessary configurations
    for (unsigned i = start_idx; i < end_idx; i++) {
      if (((i != sp_idx) || (!sp_notDominated)) &&
          ((i != en_idx) || (!en_notDominated)))
        Indices.push_back(i);
    }

    // Continue from next accuracy loss level
    start_idx = end_idx;
  }


}

void RuntimeController::compute3DParetoConfigurationPoints() {

  // Sort the configurations according to accuracy loss
  INFO("Sorting autotuner configurations...\n");
  std::sort(InitialConfigurations.begin(), InitialConfigurations.end(),
            ConfigurationLessThan());
  INFO("Done sorting.\n");

  for (unsigned start_idx = 0; start_idx < InitialConfigurations.size();) {
    // Points to first Configuration with different (higher) accuracy loss
    // compared to the one pointed by start_idx
    unsigned end_idx = start_idx + 1;
    while ((end_idx < InitialConfigurations.size()) &&
           (InitialConfigurations[end_idx].accuracyLoss -
                InitialConfigurations[start_idx].accuracyLoss <
            AL_THRESHOLD)) {
      end_idx++;
    }
    DEBUG("start_idx = %d, end_idx = %d\n", start_idx, end_idx);
    // Now, all elements in [start_idx, end_idx) have equal accuracy loss,
    // that is lower from later ones and worse than those already in curve
    // (so they cannot displace them).

    // Find candidates from [start_idx, end_idx) to be inserted
    // Keep their indices. If a point is dominated (strictly worse),
    // its index will not be inserted
    std::vector<unsigned> Indices;

    for (unsigned i = start_idx; i < end_idx; i++) {
      bool dominated = false;
      for (unsigned j = i + 1; (j < end_idx) && !dominated; j++) {
        if ((InitialConfigurations[i].speedup <
             InitialConfigurations[j].speedup) &&
            (InitialConfigurations[i].energy <
             InitialConfigurations[j].energy)) {
          dominated = true;
        }
      }
      if (!dominated) {
        DEBUG("accuracy loss = %f, speedup = %f, energy = %f, at idx = %d\n",
              InitialConfigurations[i].accuracyLoss,
              InitialConfigurations[i].speedup, InitialConfigurations[i].energy,
              i);
        Indices.push_back(i);
      }
    }

    for (std::vector<unsigned>::iterator idx_it = Indices.begin(),
                                         idx_e = Indices.end();
         idx_it != idx_e; ++idx_it) {
      Configuration &CandidateConfiguration = InitialConfigurations[*idx_it];

      if (!ThreeDCurveConfigurations.empty()) {
        bool notDominated = true;
        for (unsigned i = 0;
             (i < ThreeDCurveConfigurations.size()) && notDominated; i++) {
          if ((CandidateConfiguration.speedup <=
               ThreeDCurveConfigurations[i]->speedup) &&
              (CandidateConfiguration.energy <=
               ThreeDCurveConfigurations[i]->energy)) {
            // This configuration is not better, in at least one characteristic,
            // compared to the existing ones in the curve.
            notDominated = false;
          }
        }
        if (notDominated) {
          ThreeDCurveConfigurations.push_back(&CandidateConfiguration);
        }
      } else {
        // If the curve is empty, we know that this is a point that must be
        // inserted. It has the best accuracy loss, and belongs here because
        // it is not dominated by any point in this accuracy range.
        ThreeDCurveConfigurations.push_back(&CandidateConfiguration);
      }
    }

    // Continue from next accuracy loss level
    start_idx = end_idx;
  }
}

void RuntimeController::printConfigurations(
    std::vector<struct Configuration> &Confs) {

  for (std::vector<struct Configuration>::iterator it = Confs.begin(),
       ie = Confs.end();
       it != ie; ++it) {
    
    it->print();
  }
}

void RuntimeController::printConfigurations(
    std::vector<struct Configuration *> &Confs) {

  for (std::vector<struct Configuration *>::iterator it = Confs.begin(),
       ie = Confs.end();
       it != ie; ++it) {
    
    (*it)->print();
  }
}


double RuntimeController::getLastSpeedup() { return g_speedup; }

void RuntimeController::setLastSpeedup(double s) { g_speedup = s; }

void RuntimeController::findNextConfiguration() {
  configurationIdx = (configurationIdx + 1) % Configurations->size();
  DEBUG("findNextConfiguration: Updated configurationIdx to %u.\n",
        configurationIdx);
}

void RuntimeController::findTargetConfiguration(float goal,
                                                enum SEARCH_KIND sk) {
  // We search in range begin(), end()-1 . It is OK to decrement end(), because
  // the configurations vector always points to one of the pareto curves, and
  // they are never empty - we have always pushed at least one configuration.

  DEBUG("findTargetConfiguration: goalVal: %f, search kind: %d.\n", goal, sk);
  std::vector<struct Configuration *>::iterator low_it;
  switch (sk) {
  case SPEEDUP: {
    // Assigning one of Pareto configs to 'Configurations' class attribute
    Configurations = &SpeedupConfigurations;
    low_it =
      std::lower_bound(Configurations->begin(), Configurations->end() - 1,
		       goal, ConfigurationLessThan_SP());
    configurationIdx = low_it - Configurations->begin();
    break;
  }
  case ENERGY: {
    Configurations = &EnergyConfigurations;
    low_it =
      std::lower_bound(Configurations->begin(), Configurations->end() - 1,
		       goal, ConfigurationLessThan_E());
    configurationIdx = low_it - Configurations->begin();
    break;
  }
  case ACCURACY_LOSS: {
    Configurations = &SpeedupConfigurations;
    low_it =
      std::lower_bound(Configurations->begin(), Configurations->end() - 1,
		       goal, ConfigurationLessThan_AL());
    if ((*low_it)->accuracyLoss > goal)
      --low_it;
    configurationIdx = low_it - Configurations->begin();
    break;
  }
  default: {
    CUSTOM_ASSERT(false && "Unknown search option for optimization target");
    ERROR("Unknown search option for optimization target.");
    abort();
  }
  }
  // After search, low_it points to the Configuration to the element with the
  // goal value or the immediately lower value if it does not exist

  DEBUG("findTargetConfiguration: Updated configurationIdx to %u.\n",
        configurationIdx);
}

/***  This routine takes as input goal (target speedup) and computes the probabilty of selecting the higher configuration 
     (one with higher than target speedup) and probability of lower configuration (config with lower than target speedup).

     Motivation: The Pareto curve often does not have a configuration providing the exact req speedup
***/
void RuntimeController::adjustTargetConfiguration(float goal) {

  DEBUG("adjustTargetConfiguration: goalVal: %f.\n\n", goal);

  pseudo_rd += 0.1f;
  // Find configuration before the selected one.
  // There is always one, unless goal is 1. Then, we would pick baseline, and
  //  both upper and lower should be the same configuration, at index 0.
  unsigned prev_conf_idx =
      configurationIdx > 0 ? configurationIdx - 1 : configurationIdx;
  // Get the two configurations' speedup, and compute the appropriate ranges
  float curr_conf_speedup = (*Configurations)[configurationIdx]->speedup;
  float prev_conf_speedup = (*Configurations)[prev_conf_idx]->speedup;

  // Computation of how far the target speedup is for lower and higher speedup config
  float sp_diff = curr_conf_speedup - prev_conf_speedup;
  float high_range = curr_conf_speedup - goal;
  float low_range = goal - prev_conf_speedup;

  // These represent how likely we are to pick the upper or lower configuration
  float high_pb = 0.0, low_pb = 0.0;
  
  if (configurationIdx == prev_conf_idx) {
    high_pb = low_pb = 1.0;
  }
  else {
    // Compute the probabitly of selection for higher config and lower config
    high_pb = low_range / sp_diff;
    low_pb = high_range / sp_diff;
  }

  DEBUG("**---- adjustTargetConfiguration: upper conf = %s with probability: "
        "%f.\n",
        ((*Configurations)[configurationIdx]->name).c_str(), high_pb);
  DEBUG("**---- adjustTargetConfiguration: lower conf = %s with probability: "
        "%f.\n\n",
        ((*Configurations)[prev_conf_idx]->name).c_str(), low_pb);

  // Select a random number from 0 to 1
  // We assign the (0..low_pb) to the lower configuration, and the (low_pb..1)
  // to the upper
  // float rd = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) ;
  // float rd = pseudo_rd;
  float rd = distr(generator);
  if (rd < low_pb) {
    // If the probability is in the low range
    configurationIdx = prev_conf_idx;
  }

  DEBUG(
      "adjustTargetConfiguration: rand: %f : Updated configurationIdx to %u.\n",
      rd, configurationIdx);
}
float RuntimeController::getGoalSpeedup() {
  return 1.0 + (rand() / (RAND_MAX / (MAX_GOAL_SPEEDUP - 1.0)));
}

double RuntimeController::getBaselineTime() { return baseline_time; }

Slowdowns *RuntimeController::getSlowdowns() { return slowdowns; }

// Functions to be inserted with initializeTensorRT and clearTensorRT
extern "C" void llvm_hpvm_initializeRuntimeController(const char *ConfigFile) {
  RC = new RuntimeController();
  RC->init(ConfigFile);
  return;
}

extern "C" void llvm_hpvm_clearRuntimeController() {
  delete RC;
  return;
}

//*** Methods to compute accuracy of a tensor by the runtime controller   ***//
uint32_t *labels_from_file = NULL;

uint32_t *hpvm_rt_readLabelsBatch_cached(const char *labels_file, int start,
                                         int end) {

  // Initialize buffer
  if (!labels_from_file) {
    FILE *file = fopen(labels_file, "rb");
    if (file == NULL) {
      ERROR("Data file %s is not found. Aborting...\n", labels_file);
      abort();
    }

    // Get number of labels
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET); // return file pointer to beginning

    // Allocate memory for labels
    labels_from_file = (uint32_t *)malloc(size);
    if (labels_from_file == NULL) {
      ERROR("Memory allocation for labels unsucessfull. Aborting...\n");
      abort();
    }

    // Copy the labels file into the allocated buffer
    size_t result = fread(labels_from_file, 1, size, file);
    if (result != size) {
      // We did not read as many elemets as there are in the file
      ERROR("Reading labels file unsucessfull. Aborting...\n");
      abort();
    }

    fclose(file);
  }

  // Return pointer to labels
  return &labels_from_file[start];
}

static float average_accuracy = 0.0;
static int num_executations = 0;


float hpvm_rt_computeAccuracy3(uint32_t *labels, void *result_ptr) {

  struct Tensor *result = (struct Tensor *) result_ptr;

  size_t batch_dim = result->dims.dim_sizes[0];
  size_t num_classes = result->dims.dim_sizes[1];
  float *data = (float *)result->host_data;
  int num_errors = 0;

  printf("batch_dim = %lu, num_classes = %lu \n", batch_dim, num_classes);

  for (int i = 0; i < batch_dim; i++) {

    int chosen = 0;
    for (int id = 1; id < num_classes; ++id) {
      //printf(" check = %f \n ",  data[i * num_classes + id]);
      if (data[i * num_classes + chosen] < data[i * num_classes + id])
        chosen = id;
    }

    if (chosen != labels[i])
      num_errors++;
  }

  float accuracy = ((batch_dim - num_errors) * 1.0 / batch_dim * 1.0) * 100.0;
  printf("****** Accuracy = %f \n\n", accuracy);

  average_accuracy = accuracy + (average_accuracy * num_executations);
  num_executations++;
  average_accuracy = average_accuracy / num_executations;

  FILE *fp = fopen("final_accuracy", "w+");
  if (fp != NULL) {

    std::ostringstream ss;
    ss << std::fixed << average_accuracy;
    std::string print_str = ss.str();

    fwrite(print_str.c_str(), 1, print_str.length(), fp);
  }

  fclose(fp);

  return accuracy;
}

// This routine is used when llvm_hpvm_invokeRtControl macro is set to llvm_hpvm_invokeRtControl_BASE 
// This is the default config selection routine - it selects the first configuration in the config-file  
extern "C" void llvm_hpvm_invokeRtControl_BASE(void *result, const char *str,
                                               int start, int end) {

  uint32_t *labels_cached = hpvm_rt_readLabelsBatch_cached(str, start, end);
  hpvm_rt_computeAccuracy3(labels_cached, result);

  // Read stats for iteration that was just completed
  double current_iteration_time = RC->getCurrentIterationComputeTime();
  double current_iteration_energy = RC->getCurrentIterationComputeEnergy();

  RC->resume_profiler();
  RC->pause_profiler();

  std::pair<double, double> pinfo = RC->get_time_energy();
  RC->reset_profiler();
  RC->addToCurrentIterationControlTime(pinfo.first);
  RC->addToCurrentIterationControlEnergy(pinfo.second);

  INFO("current iteration time = %f, current iteration energy = %f\n\n",
       current_iteration_time, current_iteration_energy);

  // Note the end of iteration
  RC->end_iteration();
}


/// This routine is used when `llvm_hpvm_invokeRtControl` macro is set to `llvm_hpvm_invokeRtControl_ADJUST_PR` 
/// This routine does probabilistic selection of configurations from the Pareto curve
extern "C" void llvm_hpvm_invokeRtControl_ADJUST_PR(void *result,
                                                    const char *str,
						    int start,
                                                    int end) {

  uint32_t *labels_cached = hpvm_rt_readLabelsBatch_cached(str, start, end);
  hpvm_rt_computeAccuracy3(labels_cached, result);

  // Read stats for iteration that was just completed
  double current_iteration_energy = RC->getCurrentIterationComputeEnergy();

  RC->resume_profiler();
  double current_iteration_time = RC->getCurrentIterationComputeTime();
  double target_speedup;

  double baseline_time = RC->getBaselineTime();
  // Relative to current configuration
  target_speedup = current_iteration_time / baseline_time;
  // Adjust to baseline
  target_speedup *= RC->getCurrentConfigurationSpeedup();
  
  RC->findTargetConfiguration(target_speedup, SPEEDUP);
  RC->adjustTargetConfiguration(target_speedup);
  RC->pause_profiler();

  std::pair<double, double> pinfo = RC->get_time_energy();
  RC->reset_profiler();
  RC->addToCurrentIterationControlTime(pinfo.first);
  RC->addToCurrentIterationControlEnergy(pinfo.second);

  RC->resume_profiler();
  RC->pause_profiler();

  std::pair<double, double> pinfo2 = RC->get_time_energy();
  RC->reset_profiler();
  RC->addToCurrentIterationConfigTime(pinfo2.first);
  RC->addToCurrentIterationConfigEnergy(pinfo2.second);
 
  INFO("current iteration time = %f, current iteration energy = %f\n",
       current_iteration_time, current_iteration_energy);
  INFO("target speedup = %lf\n\n", target_speedup);

  // Note the end of iteration
  RC->end_iteration();
}


