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

#include <iostream>
#include <sstream>
#include <signal.h>
#include <cstdarg>

#include "Tools.h"
#ifdef BLAS_CUDA
# include <cuda_runtime_api.h>
#endif

extern void GpuUnlock();	// forward declaration

void Error(void)
{
  GpuUnlock();
  exit(1);
}

//******************************************

void Error(const char *txt)
{
  ErrorN(txt);
}

//******************************************

void Error(const char *txt, int ipar)
{
  fprintf(stderr,"ERROR: ");
  fprintf(stderr,txt,ipar);
  fprintf(stderr,"\n");
  GpuUnlock();
  exit(1);
}

//******************************************

__attribute__((noreturn))
void vErrorN(const char* msg, va_list args)
{
    char message[ERROR_MSG_SIZE];

#if !defined(ULTRIX) && !defined(_MINGW_) && !defined(WIN32)
    vsnprintf(message, ERROR_MSG_SIZE,msg,args);
#else
    vsprintf(message,msg,args);
#endif
    cerr <<" ERROR: "<<message<<endl;
#ifdef BLAS_CUDA
    int dev;
    cudaGetDevice(&dev);
    cerr <<" Current GPU device: "<<dev<<endl;
#endif
    GpuUnlock();
    exit(1);
}
     
//******************************************

void ErrorN(const char* msg, ...){
    va_list args;
    va_start(args,msg);
    vErrorN(msg, args);
    va_end(args);
}

//******************************************

/**
 * parses parameters written in one line like "param1=val1 param2=val2"
 * @param isInput input stream
 * @param voParams out vector of parameters
 */
void ParseParametersLine(istream& isInput, vector<boost::program_options::option>& voParams)
{
  string sRead;
  short iReadStep = 0;
  vector<boost::program_options::option>::iterator iParamIter;
  do {
    sRead.clear();
    if (iReadStep < 3)
      // read next token (name, equal character or start of value)
      isInput >> sRead;
    else {
      // read end of value in quotes
      stringbuf sbRead;
      isInput.get(sbRead, '\"');
      if (isInput.peek() != char_traits<char>::eof())
        sbRead.sputc(isInput.get());
      sRead = sbRead.str();
    }
    if (sRead.empty() || ('#' == sRead[0]) || isInput.bad())
      // stop in case of no more data, start of comment or stream error
      break;
    size_t stPos = 0;
    size_t stLen = sRead.length();

    // read equal character
    if (iReadStep == 1) {
      if (sRead[stPos] == '=') {
        stPos++;
        iReadStep = 2; // next step: read parameter value
        if (stPos >= stLen)
          continue;
      }
      else
        iReadStep = 0; // next step: read new option
    }

    // read parameter name
    if (iReadStep <= 0) {
      size_t stNPos = sRead.find('=');
      iParamIter = voParams.insert(voParams.end(), boost::program_options::option());
      iParamIter->string_key = sRead.substr(stPos, stNPos - stPos);
      iParamIter->value.push_back(string());
      stPos = stNPos;
      if (stPos != string::npos) {
        stPos++;
        iReadStep = 2; // next step: read parameter value
        if (stPos >= stLen)
          continue;
      }
      else {
        iReadStep = 1; // next step: read equal character
        continue;
      }
    }

    // read parameter value
    if (iReadStep == 2) {
      iReadStep = 0; // next loop: read new option (if value is not in quotes)
      size_t stNPos = stLen;
      if (sRead[stPos] == '\"') { // parameter value in quotes
        stPos++;
        if ((stPos < stLen) && (sRead[stLen - 1] == '\"'))
          stNPos--;
        else
          iReadStep = 3; // next loop: end reading value in quotes
      }
      iParamIter->value.back() = sRead.substr(stPos, stNPos - stPos);
      continue;
    }

    // end reading value in quotes
    if (iReadStep >= 3) {
      iParamIter->value.back().append(sRead, stPos, stLen - stPos - 1);
      iReadStep = 0; // next step: read new parameter
    }
  } while (!(sRead.empty() || isInput.bad()));
}

//******************************************

int ReadInt(ifstream &inpf, const string &name, int minval,int maxval)
{
  string buf;
  inpf >> buf;
  if (buf!=name)
    ErrorN("FileRead: found field '%s' while looking for '%s'", buf.c_str(), name.c_str());
    
  int val;
  inpf >> val;
  if (val<minval || val>maxval)
    ErrorN("FileRead: values for %s must be in [%d,%d]", name.c_str(), minval, maxval);

  return val;
}

//******************************************

#ifdef BLAS_CUDA
void DebugMachInp(string txt, REAL *iptr, int idim, int odim, int eff_bsize) {
  Error("debugging of input data not supported for CUDA"); }
void DebugMachOutp(string txt, REAL *optr, int idim, int odim, int eff_bsize) {
  Error("debugging of output data not supported for CUDA"); }
#else
void DebugMachInp(string txt, REAL *iptr, int idim, int odim, int eff_bsize)
{
  cout <<"\n" << txt;
  printf(" %dx%d bs=%d: input\n",idim,odim,eff_bsize);
  for (int bs=0;bs<eff_bsize;bs++) {
    printf("%3d:  ",bs);
    for (int i=0;i<idim;i++) printf(" %4.2f", *iptr++);
    printf("\n");
  }
}

//******************************************

void DebugMachOutp(string txt, REAL *optr, int idim, int odim, int eff_bsize)
{
  cout <<"\n" << txt;
  printf(" %dx%d bs=%d: output\n",idim,odim,eff_bsize);
  for (int bs=0;bs<eff_bsize;bs++) {
    printf("%3d:  ",bs);
    for (int o=0;o<odim;o++) printf(" %4.2f", *optr++);
    printf("\n");
  }
}
#endif
