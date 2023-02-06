#pragma once

#include <sys/time.h>

inline double time_double() {
  struct timeval now;
  gettimeofday(&now, NULL);
  double tm = (now.tv_sec * 1000000 + now.tv_usec);
  return tm / 1000000.0;
}