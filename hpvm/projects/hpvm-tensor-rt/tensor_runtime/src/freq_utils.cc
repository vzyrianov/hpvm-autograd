

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *available_freqs[] = {
    "140250000",  "229500000",  "318750000",  "408000000", "497250000",
    "586500000",  "675750000",  "765000000",  "854250000", "943500000",
    "1032750000", "1122000000", "1211250000", "1300500000"};

void updateJetsonGPUFreq(int freq_level) {

  if (freq_level < 0 || freq_level > 13) {
    printf("ERRROR: Provide freq level between {0, 13}  \n\n\n");
    abort();
  }

  const char *freq_val = available_freqs[freq_level];
  printf("freq-val[0] = %s \n", freq_val);

  FILE *max_file = fopen(
      "/sys/devices/17000000.gp10b/devfreq/17000000.gp10b/max_freq", "w+");

  if (max_file == NULL) {
    printf("Could not min_freq file \n");
  }

  fwrite(freq_val, strlen(freq_val), 1, max_file);

  fclose(max_file);

  FILE *min_file = fopen(
      "/sys/devices/17000000.gp10b/devfreq/17000000.gp10b/min_freq", "w+");

  if (min_file == NULL) {
    printf("Could not min_freq file \n");
    abort();
  }

  fwrite(freq_val, strlen(freq_val), 1, min_file);

  fclose(min_file);
}

unsigned long int readJetsonGPUFreq() {

  FILE *cur_freq_file =
      fopen("/sys/devices/17000000.gp10b/devfreq/17000000.gp10b/cur_freq", "r");

  if (cur_freq_file == NULL) {
    printf("Could not open cur_freq file \n");
  }

  char buf[50];
  char *ptr;

  fread(buf, 50, 1, cur_freq_file);

  unsigned long cur_freq = strtoul(buf, &ptr, 10);

  fclose(cur_freq_file);

  return cur_freq;
}

int main() {

  updateJetsonGPUFreq(7);

  unsigned long int cur_freq = readJetsonGPUFreq();

  printf("** cur_freq = %lu \n\n", cur_freq);

  return 0;
}
