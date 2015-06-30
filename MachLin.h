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
 *
 * linear machine:  output = weights * input + biases
 */

#ifndef _MachLin_h
#define _MachLin_h

#include <pthread.h>
#include "Mach.h"
#include "Shareable.h"

class MachLin : public Mach, public Shareable
{
private:
  void do_alloc();	// perform allocation of dynamic data structures
protected:
  int  nb_params;	// number of params for max-out
   // CUDA: the following two variables refer to device memory
  REAL *b;		// biases
  REAL *w;		// weights, stored in BLAS format, e.g. COLUMN major !
  REAL clip_w;		// absolute value for clipping weights, default=0, i.e. no clipping
  REAL clip_gradw;	// absolute value for clipping gradients on weights, default=0, i.e. no clipping
  REAL clip_gradb;	// absolute value for clipping gradients on biaises, default=0, i.e. no clipping
  int *bw_shared;	// number of objects sharing biases and weights
  pthread_mutex_t *bw_mutex;	// mutex used to share biases and weights
  virtual void ReadParams(istream&, bool=true); // read all params
  virtual void ReadData(istream&, size_t, int=0); // read binary data
  virtual void WriteParams(ostream&); // write all params
  virtual void WriteData(ostream&); // write binary data
  MachLin(const MachLin &);			// create a copy of the machine, sharing the parameters
public:
  //MachLin(const int=0, const int=0, const int=128, const ulong=0, const ulong=0);	
  MachLin(const int p_idim=0, const int p_odim=0, const int p_bsize=128, const ulong p_nbfw=0, const ulong p_nbbw=0, const int shareid=-1, const bool xdata=false);
  //MachLin(const int=0, const int=0, const int=128, const ulong=0, const ulong=0, const int=-1, const bool=false);	
  virtual ~MachLin();
  virtual MachLin *Clone() {return new MachLin(*this);}		// create a copy of the machine, sharing the parameters
  virtual ulong GetNbParams() {return idim*odim+odim;}		// return the nbr of allocated parameters 
  virtual int GetMType() {return file_header_mtype_lin;};	// get type of machine
    // set values for clipping
  virtual void SetClipW(REAL v) {clip_w=v;};			
  virtual void SetClipGradW(REAL v) {clip_gradw=v;};			
  virtual void SetClipGradB(REAL v) {clip_gradb=v;};			
    // network initialisation
  virtual void BiasConst(const REAL val);			// init biases with constant values
  virtual void BiasRandom(const REAL range);			// random init of biases in [-range, range]
  virtual void WeightsConst(const REAL val);			// init weights with constant values
  virtual void WeightsID(const REAL =1.0);			// init weights to identity transformation
  virtual void WeightsRandom(const REAL range);			// random init of weights in [-range, range]
  virtual void WeightsRandomFanI(const REAL range=sqrt(6.0));	// random init of weights in fct of fan-in
  virtual void WeightsRandomFanIO(const REAL range=sqrt(6.0));	// random init of weights in fct of fan-in and fan-out
  virtual void Info(bool=false, char *txt=(char*)"");		// display (detailed) information on machine
  virtual bool CopyParams(Mach*);	// copy parameters from another machine
  virtual void Forw(int=0, bool=false);	// calculate outputs for current inputs
  virtual void ForwDropout(int=0, bool=false);	// new function to apply dropout in training forward pass, must be called AFTER output function
    // backprop gradients from output to input and update all weights
  virtual void Backw (const float lrate, const float wdecay, int=0);
  virtual void Debug ();
};

#endif
