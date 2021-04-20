

#include <fstream>
#include <map>
#include <sstream>
#include <string.h>
#include <vector>

#include "approx_knob_utils.h"
#include "debug.h"
#include "config.h"

PerfParams::PerfParams() {
  row = 1;
  col = 1;
  skip_offset = 0;
}

PerfParams::PerfParams(int row1, int col1, int skip_offset1) {
  row = row1;
  col = col1;
  skip_offset = skip_offset1;
}

PerfParamSet::PerfParamSet() {

  printf("- knobs_file_path = %s \n", GLOBAL_KNOBS_FILE);
  std::ifstream file(GLOBAL_KNOBS_FILE);

  if (!file) {
    ERROR(" Could NOT find global_knobs.txt \n");
  }

  std::string line;
  std::string partial;
  std::vector<std::string> tokens;

  while (std::getline(file, line)) { // Read each line

    // printf ("***** line === %s ", line);
    std::istringstream iss(line);
    std::string token;
    while (std::getline(iss, token, '\t')) { // Read each token in the line
      tokens.push_back(token);

      int index = token.find("perf");
      if (index != std::string::npos) {

        int index2 = token.find(",");
        std::string knob_str = token.substr(index2 + 1);
        int knob = atoi(knob_str.c_str());

        std::getline(iss, token, '\t');
        std::istringstream token_stream(token);

        std::string tok;

        std::getline(token_stream, tok, ',');
        int row = atoi(tok.c_str());

        std::getline(token_stream, tok, ',');
        int col = atoi(tok.c_str());

        std::getline(token_stream, tok, ',');
        int offset = atoi(tok.c_str());

        // printf("**** knob = %d, row = %d, col = %d, offset = %d \n\n", knob,
        //       row, col, offset);
        PerfParams params(row, col, offset);
        perf_knob_map[knob] = params;
      }
    }
  }

  file.close();
}

PerfParams PerfParamSet::getPerfParams(int swing) {

  if (swing >= 150) {
    swing = swing - 30;
  }

  return perf_knob_map[swing];
}

SampParams::SampParams() {
  skip_rate = 1;
  skip_offset = 0;
}

SampParams::SampParams(int skip_rate1, int skip_offset1,
                       float interpolation_id1) {
  skip_rate = skip_rate1;
  skip_offset = skip_offset1;
  interpolation_id = interpolation_id1;
}

SampParamSet::SampParamSet() {

  printf("- knobs_file_path = %s \n", GLOBAL_KNOBS_FILE);
  std::ifstream file(GLOBAL_KNOBS_FILE);

  if (!file) {
    ERROR("Could NOT find global_knobs.txt \n");
  }

  std::string line;
  std::string partial;
  std::vector<std::string> tokens;

  while (std::getline(file, line)) { // Read each line

    std::istringstream iss(line);
    std::string token;
    while (std::getline(iss, token, '\t')) { // Read each token in the line
      tokens.push_back(token);

      int index = token.find("samp");
      int test_index = token.find("reduction");

      if (index != std::string::npos && test_index == std::string::npos) {

        int index2 = token.find(",");
        std::string knob_str = token.substr(index2 + 1);
        int knob = atoi(knob_str.c_str());
        // printf("knob = %d \n", knob);

        std::getline(iss, token, '\t');
        std::istringstream token_stream(token);

        std::string tok;

        std::getline(token_stream, tok, ',');
        int skip_every = atoi(tok.c_str());

        std::getline(token_stream, tok, ',');
        int offset = atoi(tok.c_str());

        std::getline(token_stream, tok, ',');
        float interpolation_id = atof(tok.c_str());

        // printf("skip_every = %d, offset = %d \n", skip_every, offset);
        SampParams params(skip_every, offset, interpolation_id);
        samp_knob_map[knob] = params;
      }
    }
  }

  file.close();
}

SampParams SampParamSet::getSampParams(int swing) {

  if (swing >= 260) {
    swing = swing - 30;
  }

  return samp_knob_map[swing];
}

RedSampParams::RedSampParams() {
  skip_ratio = 0.0f;
  is_half = false;
}

RedSampParams::RedSampParams(float skip_ratio1, bool is_half1) {
  skip_ratio = skip_ratio1;
  is_half = is_half1;
}

RedSampParams getRedSampParams(int swing) {

  std::map<int, RedSampParams> red_samp_knob_map;

  RedSampParams params41(0.5, false);
  red_samp_knob_map[41] = params41;

  RedSampParams params42(0.5, true);
  red_samp_knob_map[42] = params42;

  RedSampParams params43(0.4, false);
  red_samp_knob_map[43] = params43;

  RedSampParams params44(0.4, true);
  red_samp_knob_map[44] = params44;

  RedSampParams params45(0.25, false);
  red_samp_knob_map[45] = params45;

  RedSampParams params46(0.25, true);
  red_samp_knob_map[46] = params46;

  return red_samp_knob_map[swing];
}
