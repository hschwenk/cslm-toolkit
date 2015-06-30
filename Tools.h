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

#ifndef _Tools_h
#define _Tools_h

#include <boost/program_options/option.hpp>
#include <iostream>
#include <istream>
#include <fstream>
#include <string.h>	// memcpy()
#include <stdlib.h>	// exit()
#include <math.h>	
#include <limits>
#include <vector>
using namespace std;

typedef float REAL;			// precision of all the calculations
typedef int WordID;			// size of the binary word indices
#define NULL_WORD (-1)			// this is used to simulate an empty input which has no effect
					// on the forward and backward pass (since we have a fixed context length)

typedef unsigned int uint;
typedef long unsigned int luint;

static const int max_word_len=65534;	// should be more than enough when reading text lines ;-)
static const string cslm_version="V3.1";

//
// general purpose helper functions
//
#ifdef DEBUG
# define TRACE(txt) cout << txt;
#else
# define TRACE(txt)
#endif

#ifdef DEBUGEX
  void DebugMachInp(string txt, REAL *iptr, int idim, int odim, int eff_bsize);
  void DebugMachOutp(string txt, REAL *optr, int idim, int odim, int eff_bsize);
# define debugMachInp(txt,adr,idim,odim,eff_bsize) DebugMachInp(txt,adr,idim,odim,eff_bsize)
# define debugMachOutp(txt,adr,idim,odim,eff_bsize) DebugMachOutp(txt,adr,idim,odim,eff_bsize)
#else
# define debugMachInp(txt,adr,idim,odim,eff_bsize)
# define debugMachOutp(txt,adr,idim,odim,eff_bsize)
#endif

#define ERROR_MSG_SIZE 4096
void Error(void);
void Error(const char *txt);
void Error(const char *txt, int);
void ErrorN(const char* msg, ...)
    __attribute__((noreturn))
    __attribute__((format(printf, 1, 2)));

#define CHECK_FILE(ifs,fname) if(!ifs) { perror(fname); Error(); }

/**
 * parses parameters written in one line like "param1=val1 param2=val2"
 * @param isInput input stream
 * @param voParams out vector of parameters
 */
void ParseParametersLine(istream& isInput, vector<boost::program_options::option>& voParams);

//
// parsing of ASCII files
//
int ReadInt(ifstream&,const string&,int=0,int=numeric_limits<int>::max());
float ReadFloat(ifstream&,const string&,float=0,float=numeric_limits<float>::max());
string ReadText(ifstream&,const string&);

//
// logarithm with bound checking for very small values
// (this prevents rounding small values to 0 which give a -NAN when taking the log)

const REAL LOG_LOWER_BOUND=2.0*numeric_limits<float>::min();	// the lower bound is somehow arbitrary, but shouldn't matter in real calculations
inline REAL safelog(REAL x) { return (x<LOG_LOWER_BOUND) ? log(LOG_LOWER_BOUND) : log(x); };

#endif
