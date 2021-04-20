
#ifndef PROFILING_HEADER
#define PROFILING_HEADER

/***** Profiling routines ***/

extern "C" {

void startProfiling();

void stopProfiling();

void profileEvent(const char *event_name, bool compare_previous = false);
}

#endif
