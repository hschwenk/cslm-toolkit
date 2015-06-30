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
 * This machines implements a set of INDEPENDENT machines which outputs are
 * combined (max, average, etc according to the subclasses). The machines must have the
 * same input and output dimension.  The order of the forward and backward
 * passes are not defined and may be in parallel on multiple CPUs or GPUs
 *
 * memory management:
 *  - data_in	same pointer for all machines
 *  - data_out	allocated (to calculate the max)
 *  - grad_in	allocated (sum of Machine's grad_in)
 *  - grad_out	points to following machine
 *  		we also allocate internal storage to set some gradients
 *  		of the individual machines to zero (this is faster than
 *  		selective vector-wise backprop)
 */

#ifndef _MachAvr_h
#define _MachAvr_h

using namespace std;
#include <vector>

#include "MachCombined.h"

class MachAvr : public MachCombined
{
private:
  void do_alloc();
protected:
  virtual void ReadData(istream&, size_t, int=0); 	// read binary data
  MachAvr(const MachAvr &);			// create a copy of the machine (without submachines)
public:
  MachAvr();	// create initial sequence with no machine
  virtual ~MachAvr();
  virtual MachAvr *Clone();			// create a copy of the machine and all submachines
  virtual int GetMType() {return file_header_mtype_avr;};	// get type of machine
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
