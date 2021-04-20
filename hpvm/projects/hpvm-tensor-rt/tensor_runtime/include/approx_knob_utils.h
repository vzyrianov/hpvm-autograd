#ifndef APPROX_KNOBS_UTILS
#define APPROX_KNOBS_UTILS

#include <fstream>
#include <map>
#include <sstream>
#include <vector>

class PerfParams {

public:
  int row;
  int col;
  int skip_offset;

  PerfParams();

  PerfParams(int row1, int col1, int skip_offset1);
};

class PerfParamSet {

private:
  std::map<int, PerfParams> perf_knob_map;

public:
  PerfParamSet();

  PerfParams getPerfParams(int knob_id);
};

class SampParams {

public:
  int skip_rate;
  int skip_offset;
  float interpolation_id;

  SampParams();

  SampParams(int skip_rate1, int skip_offset1, float interpolation_id1);
};

class SampParamSet {

private:
  std::map<int, SampParams> samp_knob_map;

public:
  SampParamSet();

  SampParams getSampParams(int knob_id);
};

class RedSampParams {

public:
  float skip_ratio;
  bool is_half;

  RedSampParams();

  RedSampParams(float skip_ratio1, bool is_half1);
};

RedSampParams getRedSampParams(int knob_id);

#endif
