
/****  

   This file contains freqency setting routines specific to the Jetson Tx2 

   NOTE: These routines are not used directly in the current code. 

   Users testing frequency changes on the Jetson Tx2 (or similar devices) can use/repurpose these routines

***/

#include <fstream>


const int available_freqs[] = {
    140250000,  // 0
    229500000,  // 1
    318750000,  // 2
    408000000,  // 3
    497250000,  // 4
    586500000,  // 5
    675750000,  // 6
    765000000,  // 7
    854250000,  // 8
    943500000,  // 9
    1032750000, // 10
    1122000000, // 11
    1211250000, // 12
    1300500000  // 13
};


// Sets frequency
void setFreq(unsigned freq_index) {

  unsigned target_freq = available_freqs[freq_index];

  const char *const min_freq_file =
      "/sys/devices/17000000.gp10b/devfreq/17000000.gp10b/min_freq";
  const char *const max_freq_file =
      "/sys/devices/17000000.gp10b/devfreq/17000000.gp10b/max_freq";

  std::ofstream min_stream;
  std::ofstream max_stream;

  min_stream.open(min_freq_file, std::ofstream::out);
  max_stream.open(max_freq_file, std::ofstream::out);

  min_stream << target_freq << std::flush;
  max_stream << target_freq << std::flush;

  min_stream.close();
  max_stream.close();
}

// Records frequency
unsigned recordFreq() {

  // Current frequency file
  const char *const cur_freq_file =
      "/sys/devices/17000000.gp10b/devfreq/17000000.gp10b/cur_freq";
  std::ifstream cur_stream;
  cur_stream.open(cur_freq_file, std::ifstream::in);

  // Get starting frequency
  unsigned cur_freq;
  cur_stream >> cur_freq;
  std::cout << "Starting frequency = " << cur_freq << "\n";
  cur_stream.close();

  return cur_freq;
}

// There will be no frequency request for the first batch
// Therefore, we skip the first element by initializing to 1, not 0.
FrequencyIndexList::FrequencyIndexList(std::vector<int> il, unsigned rf)
    : idx_list(il), rep_factor(rf), count(1), idx(0) {}

unsigned FrequencyIndexList::getNextIndex() {
  if (count == rep_factor) {
    count = 0;
    idx = (idx + 1) % idx_list.size();
  }
  count++;
  return idx_list[idx];
}


void RuntimeController::readIterationFrequency() {
  if (PI)
    PI->readIterationFrequency();
}

unsigned long RuntimeController::getIterationFrequency() {
  return (PI ? PI->getIterationFrequency() : 0);
}

void RuntimeController::updateFrequency() {
#ifdef JETSON_EXECUTION
  unsigned freq_idx = FIL->getNextIndex();
  //--- updateJetsonGPUFreq(freq_idx);

  setFreq(freq_idx);

#endif // JETSON_EXECUTION
}

unsigned long RuntimeController::getLastFrequency() { return g_freq; }

void RuntimeController::setLastFrequency(unsigned long f) { g_freq = f; }



void ProfileInfo::readIterationFrequency() {
#ifdef JETSON_EXECUTION
  //----- frequency_current_iteration = readJetsonGPUFreq();
  frequency_current_iteration = recordFreq();
#else
  frequency_current_iteration = 0;
#endif // JETSON_EXECUTION
}

unsigned long ProfileInfo::getIterationFrequency() {
  return frequency_current_iteration;
}
