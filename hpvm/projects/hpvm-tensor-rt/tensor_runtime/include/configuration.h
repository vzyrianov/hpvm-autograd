#ifndef LLVM_HPVM_CONFIGURATION_H
#define LLVM_HPVM_CONFIGURATION_H

#include <map>
#include <vector>

#include "debug.h"

// Configuration related class definitions - in Configuration.h

// Describes the internal choices made for an ApproxHPVM node
class NodeConfiguration {
public:
  enum NODE_CONFIGURATION_TARGET { GPU, CPU, END };

protected:
  enum NODE_CONFIGURATION_TARGET NODE_CONFIGURATION_TARGET_ID;

public:
  bool isGPUNodeConfiguration();

  bool isCPUNodeConfiguration();

  virtual void print() = 0;
};

class GPUNodeConfiguration : public NodeConfiguration {
public:
  // Approximation methods available for this HW type
  enum APPROX {
    FP32,
    FP16,
    PERFORATION,
    PERFORATION_HP,
    INPUT_SAMPLING,
    INPUT_SAMPLING_HP,
    REDUCTION_SAMPLING,
    //  ADDITIONAL_APPROXIMATION_METHOD
    APPROX_END
  };

  // Operations to be approximated in the node using this configuration
  enum TENSOR_OP {
    ADD,
    BATCHNORM,
    CONV,
    GROUP_CONV,
    MUL,
    RELU,
    CLIPPED_RELU,
    TANH,
    POOL_MAX,
    POOL_MEAN,
    POOL_MIN,
    SOFTMAX,
    // ADDITIONAL_TENSOR_OPERATION
    TENSOR_OP_END
  };

private:
  // A vector, containing pairs of approximation method and tunable parameter
  // (expressed as int, or ignored when not applicable) for each operation
  std::vector<
      std::pair<enum TENSOR_OP, std::vector<std::pair<enum APPROX, int>>>>
      ApproxChoices;

public:
  void pushNewTensorOperation(enum TENSOR_OP top);

  void pushNewApproximationChoiceForOperation(enum APPROX approx, int u);

  std::vector<
      std::pair<enum TENSOR_OP, std::vector<std::pair<enum APPROX, int>>>> &
  getApproxChoices();

  GPUNodeConfiguration();
  ~GPUNodeConfiguration();

  void print() override;
};

class CPUNodeConfiguration : public NodeConfiguration {
public:
  // Approximation methods available for this HW type
  enum APPROX {
    FP32,
    PERFORATION,
    INPUT_SAMPLING,
    //  ADDITIONAL_APPROXIMATION_METHOD
    APPROX_END
  };

  // Operations to be approximated in the node using this configuration
  enum TENSOR_OP {
    ADD,
    BATCHNORM,
    CONV,
    GROUP_CONV,
    MUL,
    RELU,
    CLIPPED_RELU,
    TANH,
    POOL_MAX,
    POOL_MEAN,
    POOL_MIN,
    SOFTMAX,
    //  ADDITIONAL_TENSOR_OPERATION
    TENSOR_OP_END
  };

private:
  // A vector, containing pairs of approximation method and tunable parameter
  // (expressed as int, or ignored when not applicable) for each operation
  std::vector<
      std::pair<enum TENSOR_OP, std::vector<std::pair<enum APPROX, int>>>>
      ApproxChoices;

public:
  void pushNewTensorOperation(enum TENSOR_OP top);

  void pushNewApproximationChoiceForOperation(enum APPROX approx, int u);

  std::vector<
      std::pair<enum TENSOR_OP, std::vector<std::pair<enum APPROX, int>>>> &
  getApproxChoices();

  CPUNodeConfiguration();
  ~CPUNodeConfiguration();

  void print() override;
};

// Configuration : Includes configuration information :
// - name
// - speedup
// - energy
// - accuracy (compared to golden output)
// - accuracy loss (compared to baseline)
// - a hardware choice and set or operations-approximation choices, described in
// setup
struct Configuration {
  std::string name;
  float speedup;
  float energy;
  float accuracy;
  float accuracyLoss;
  std::map<std::string, NodeConfiguration *> setup;
  // map for mapping visc.node.id IDs to HPVM (fused) node approx-configurations
  std::map<int, NodeConfiguration *> idConfigMap;

  Configuration(std::string &n, float f, float e, float a, float al);

  float getSpeedup();

  float getEnergy();

  float getAccuracy();

  float getAccuracyLoss();

  void print();
};

// Comparison operator definition, in increasing accuracy loss
// (for std sort, used in pareto optimal computation)
struct ConfigurationLessThan {
  bool operator()(const struct Configuration &a,
                  const struct Configuration &b) const;
};

// Comparison operator definition, in increasing accuracy loss
// (for std lower bound, used in pareto optimal frontier search)
struct ConfigurationLessThan_AL {
  bool operator()(const struct Configuration *a, const float &b) const;
};

// Comparison operator definition, in increasing speedup
// (for std lower bound, used in pareto optimal frontier search)
struct ConfigurationLessThan_SP {
  bool operator()(const struct Configuration *a, const float &b) const;
};

// Comparison operator definition, in decreasing energy
// (for std lower bound, used in pareto optimal frontier search)
struct ConfigurationLessThan_E {
  bool operator()(const struct Configuration *a, const float &b) const;
};

enum SEARCH_KIND { SPEEDUP, ENERGY, ACCURACY_LOSS, END };

#endif
