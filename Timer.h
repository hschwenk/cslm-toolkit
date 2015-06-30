/*
 * This file is part of the continuous space language and translation model toolkit
 * for statistical machine translation and large vocabulary speech recognition.
 *
 * Copyright 2015, Holger Schwenk, LIUM, University of Le Mans, France
 *
 * The CSLM toolkit is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License version 3 as
 * published by the Free Software Foundation
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
 *
 *
 */

#ifndef _Timer_h
#define _Timer_h

#define PROFILE

#ifdef PROFILE

#include <cstdio>
#include <ctime>
#include <sys/time.h>

class Timer {
private:
  clock_t c_cumul;	// cumulated CPU clocks
  double r_cumul;	// cumulated real time
  clock_t c_beg;
  timeval t_beg;
public:
  Timer() : c_cumul(0), r_cumul(0) {}
  void start() {
    c_beg=clock();
    gettimeofday(&t_beg, NULL);
  }
  void stop() {
    c_cumul += clock()-c_beg;

    timeval t_end;
    gettimeofday(&t_end, NULL);
    r_cumul += (double) (t_end.tv_sec + t_end.tv_usec/1000000.0)
             - (double) (t_beg.tv_sec + t_beg.tv_usec/1000000.0);
  }

  void disp(const char *txt) {
    if (r_cumul>0) printf("%sreal=%.2fs, cpu=%.2fs", txt, r_cumul, (float) c_cumul/CLOCKS_PER_SEC);
  }
  void newline() { printf("\n"); }
};

#else // globally deactivate profiling, nothing is counted or printed

class Timer {
private:
public:
  Timer() {};
  void start() {};
  void stop() {};
  void disp(const char *txt) {};
  void newline() {};
};

#endif

#endif //_Timer_h
