//===--------------------------- configuration.cpp
//-------------------------===//
//
//===----------------------------------------------------------------------===//
//
//  This file  consists of the definitions of API to get information about
// configurations for rest of the tensor runtime to use.
//
//===----------------------------------------------------------------------===//

#include "configuration.h"
#include <algorithm>
#include <string>

using G_APPROX = GPUNodeConfiguration::APPROX;
using C_APPROX = CPUNodeConfiguration::APPROX;
using G_TENSOR_OP = GPUNodeConfiguration::TENSOR_OP;
using C_TENSOR_OP = CPUNodeConfiguration::TENSOR_OP;

bool NodeConfiguration::isGPUNodeConfiguration() {
  return NODE_CONFIGURATION_TARGET_ID == GPU;
}

bool NodeConfiguration::isCPUNodeConfiguration() {
  return NODE_CONFIGURATION_TARGET_ID == CPU;
}

void GPUNodeConfiguration::pushNewTensorOperation(G_TENSOR_OP top) {
  std::vector<std::pair<G_APPROX, int>> emptyVec;
  ApproxChoices.push_back(std::make_pair(top, emptyVec));
}

void GPUNodeConfiguration::pushNewApproximationChoiceForOperation(
    G_APPROX approx, int u) {
  unsigned size = ApproxChoices.size();
  CUSTOM_ASSERT(size >= 1 &&
                "Cannot apply approximation choice to non existent operation.");
  ApproxChoices[size - 1].second.push_back(std::make_pair(approx, u));
}

std::vector<std::pair<G_TENSOR_OP, std::vector<std::pair<G_APPROX, int>>>> &
GPUNodeConfiguration::getApproxChoices() {
  return ApproxChoices;
}

GPUNodeConfiguration::GPUNodeConfiguration() {
  NODE_CONFIGURATION_TARGET_ID = GPU;
}
GPUNodeConfiguration::~GPUNodeConfiguration() {}

void CPUNodeConfiguration::pushNewTensorOperation(C_TENSOR_OP top) {
  std::vector<std::pair<C_APPROX, int>> emptyVec;
  ApproxChoices.push_back(std::make_pair(top, emptyVec));
}

void CPUNodeConfiguration::pushNewApproximationChoiceForOperation(
    C_APPROX approx, int u) {
  unsigned size = ApproxChoices.size();
  CUSTOM_ASSERT(size >= 1 &&
                "Cannot apply approximation choice to non existent operation.");
  ApproxChoices[size - 1].second.push_back(std::make_pair(approx, u));
}

std::vector<std::pair<C_TENSOR_OP, std::vector<std::pair<C_APPROX, int>>>> &
CPUNodeConfiguration::getApproxChoices() {
  return ApproxChoices;
}

CPUNodeConfiguration::CPUNodeConfiguration() {
  NODE_CONFIGURATION_TARGET_ID = CPU;
}
CPUNodeConfiguration::~CPUNodeConfiguration() {}

Configuration::Configuration(std::string &n, float f, float e, float a,
                             float al)
    : name(n), speedup(f), energy(e), accuracy(a), accuracyLoss(al) {}

float Configuration::getSpeedup() { return speedup; }

float Configuration::getEnergy() { return energy; }

float Configuration::getAccuracy() { return accuracy; }

float Configuration::getAccuracyLoss() { return accuracyLoss; }
bool ConfigurationLessThan::operator()(const struct Configuration &a,
                                       const struct Configuration &b) const {
  return (a.accuracyLoss < b.accuracyLoss);
}
bool ConfigurationLessThan_AL::operator()(const struct Configuration *a,
                                          const float &b) const {
  return (a->accuracyLoss < b);
}
bool ConfigurationLessThan_SP::operator()(const struct Configuration *a,
                                          const float &b) const {
  return (a->speedup < b);
}
bool ConfigurationLessThan_E::operator()(const struct Configuration *a,
                                         const float &b) const {
  return (a->energy < b);
}

//****** HEADER Ends - Source Starts

// Helper configuration print methods

void GPUNodeConfiguration::print() {

  printf(" gpu");
  for (auto &it : ApproxChoices) {

    printf(" ");
    switch (it.first) {
    case G_TENSOR_OP::ADD:
      printf("add");
      break;
    case G_TENSOR_OP::BATCHNORM:
      printf("batchnorm");
      break;
    case G_TENSOR_OP::CONV:
      printf("conv");
      break;
    case G_TENSOR_OP::GROUP_CONV:
      printf("group_conv");
      break;
    case G_TENSOR_OP::MUL:
      printf("mul");
      break;
    case G_TENSOR_OP::RELU:
      printf("relu");
      break;
    case G_TENSOR_OP::CLIPPED_RELU:
      printf("clipped_relu");
      break;
    case G_TENSOR_OP::TANH:
      printf("tanh");
      break;
    case G_TENSOR_OP::POOL_MAX:
      printf("pool_max");
      break;
    case G_TENSOR_OP::POOL_MEAN:
      printf("pool_mean");
      break;
    case G_TENSOR_OP::POOL_MIN:
      printf("pool_min");
      break;
    case G_TENSOR_OP::SOFTMAX:
      printf("softmax");
      break;
      // TODO additional operations to be printed here
    default:
      ERROR("Unknown tensor operation.");
      break;
    }

    auto &approxVec = it.second;
    for (auto &inner_it : approxVec) {
      printf(" ");
      switch (inner_it.first) {
      case G_APPROX::FP32:
        printf("fp32");
        break;
      case G_APPROX::FP16:
        printf("fp16");
        break;
      case G_APPROX::PERFORATION:
        printf("perf");
        break;
      case G_APPROX::PERFORATION_HP:
        printf("perf_fp16");
        break;
      case G_APPROX::INPUT_SAMPLING:
        printf("samp");
        break;
      case G_APPROX::INPUT_SAMPLING_HP:
        printf("samp_fp16");
        break;
      case G_APPROX::REDUCTION_SAMPLING:
        printf("red_samp");
        break;
        // TODO additional approx methods to be printed here
      default:
        ERROR("Unknown approximation option");
        break;
      }

      printf(" %d", inner_it.second);
    }
  }

  printf("\n");
}

void CPUNodeConfiguration::print() {

  printf(" cpu");
  for (auto &it : ApproxChoices) {

    printf(" ");
    switch (it.first) {
    case C_TENSOR_OP::ADD:
      printf("add");
      break;
    case C_TENSOR_OP::BATCHNORM:
      printf("batchnorm");
      break;
    case C_TENSOR_OP::CONV:
      printf("conv");
      break;
    case C_TENSOR_OP::GROUP_CONV:
      printf("group_conv");
      break;
    case C_TENSOR_OP::MUL:
      printf("mul");
      break;
    case C_TENSOR_OP::RELU:
      printf("relu");
      break;
    case C_TENSOR_OP::CLIPPED_RELU:
      printf("clipped_relu");
      break;
    case C_TENSOR_OP::TANH:
      printf("tanh");
      break;
    case C_TENSOR_OP::POOL_MAX:
      printf("pool_max");
      break;
    case C_TENSOR_OP::POOL_MEAN:
      printf("pool_mean");
      break;
    case C_TENSOR_OP::POOL_MIN:
      printf("pool_min");
      break;
    case C_TENSOR_OP::SOFTMAX:
      printf("softmax");
      break;
      // TODO additional operations to be printed here
    default:
      ERROR("Unknown tensor operation.");
      break;
    }

    auto &approxVec = it.second;
    for (auto &inner_it : approxVec) {
      printf(" ");
      switch (inner_it.first) {
      case C_APPROX::FP32:
        printf("fp32");
        break;
      case C_APPROX::PERFORATION:
        printf("perf");
        break;
      case C_APPROX::INPUT_SAMPLING:
        printf("samp");
        break;
        // TODO additional approx methods to be printed here
      default:
        ERROR("Unknown approximation option");
        break;
      }

      printf(" %d", inner_it.second);
    }
  }

  printf("\n");
}


struct NodeConfigID{

  int hpvm_id;
  NodeConfiguration* node_config;

  bool operator < (const NodeConfigID& obj) const
  {
    return (hpvm_id < obj.hpvm_id);
  }
};



void Configuration::print() {

  std::vector<NodeConfigID> config_vector;
  for (std::map<std::string, NodeConfiguration *>::const_iterator it =
	 setup.begin();
       it != setup.end(); ++it) {

    NodeConfigID node;
    node.hpvm_id = std::stoi(it->first);
    node.node_config = it->second;
    config_vector.push_back(node);
  }
 
  std::sort(config_vector.begin(), config_vector.end());
  
  printf("+++++\n");
  printf("%s %f %f %f %f\n", name.c_str(), speedup, energy, accuracy,
         accuracyLoss);
  for (unsigned int i = 0; i < config_vector.size(); i++){

    NodeConfigID conf = config_vector[i];
    printf(" %d : ", conf.hpvm_id);
    conf.node_config->print();
  }  
  printf("-----\n");
  
}

