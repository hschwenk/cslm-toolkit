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
 *  Join machine:
 *   - combines several machines into one layer
 *   - the output dimensions must be identical, the input dimensions may differ
 */

#ifndef _MachJoin_h
#define _MachJoin_h

using namespace std;
#include <vector>

#include "MachMulti.h"

class MachJoin : public MachMulti
{
private:
#ifdef BLAS_CUDA
  REAL* gpu_dev_data_out;	// local copy of output buffer
  REAL* sub_input_tmp;			// Temporarily hold the input of a sub-machine before transfer to the right device
#else
  REAL* grad_out_copy;	// copy of output gradients, that is passed to the sub-machines' Backw()
#endif
  void do_alloc(bool);	// perform allocation of dynamic data structures
  void do_delete();	// delete data structures
protected:
  virtual void ReadData(istream&, size_t, int=0); // read binary data
  MachJoin(const MachJoin &);			// create a copy of the machine (without submachines)
public:
  MachJoin();	// create initial sequence with no machine
  virtual ~MachJoin();
  virtual MachJoin *Clone();			// create a copy of the machine and all submachines
  virtual int GetMType() {return file_header_mtype_mjoin;};	// get type of machine
    // redfine connecting functions
  virtual void SetDataIn(REAL*);	// set pointer of input data
  virtual void SetGradOut(REAL*);	// set pointer of output gradient 
    // add and remove machines
  virtual void MachAdd(Mach*); // add new machine after the existing ones
  virtual Mach *MachDel();
    // standard functions
  virtual void Info(bool=false, char *txt=(char*)"");	// display (detailed) information on machine
  virtual void Forw(int=0, bool=false);	// calculate outputs for current inputs
  virtual void Backw(const float lrate, const float wdecay, int=0);	// calculate gradients at input for current gradients at output
};

#endif
