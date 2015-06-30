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
 * Split one input into several output layers:
 *   - all the indiv machines must have the same input dimension and bsize
 *   - the output dimension can vary
 *   - the output and gradient of the split machines are independent
 *     it is the responsibility of the Trainer to provide gradients for each one !
 *
 * Dimensions:
 *   - idim: same for all machines
 *   - odim: sum of output dimensions of all machines
 *   	     There are no common data structures of this dimension !!
 *   - bsize: identical for all machines
 *
 * Memory allocation:
 *  - output data:  each individual machine has its own self-allocated output layer
 *  		    the output of the overall split machine is set to NULL
 *  - input data:   all indiv machines share the same pointer on the input
 *  		    (not allocated by the split machine)
 *  - input gradient: cumulates gradients of indiv. machines
 *  		    allocated by the split machine
 *
 *   Difference to MachSplit1()
 *     MachSPlit1() uses one concatenated output and gradient layer
 *     However, this involves costly copying of data, in particular on GPUs
 */

#ifndef _MachSplit_h
#define _MachSplit_h

using namespace std;
#include <vector>

#include "MachMulti.h"

class MachSplit : public MachMulti
{
private:
  Timer tbackw;
#ifdef BLAS_CUDA
					// local device specific copies of data for each GPU (needed for all devices different from master)
  vector<REAL*> gpu_dev_data_in;	// local copy of input buffers
  vector<REAL*> gpu_dev_grad_in;	// local copy of input gradients
					// these buffers are allocated once when first machine is added
					// They do not change when other machines are added!
#endif
  void do_alloc();	// perform allocation of dynamic data structures
  void do_delete();	// delete data structures
protected:
  virtual void ReadData(istream&, size_t, int=0); // read binary data
  MachSplit(const MachSplit &);			// create a copy of the machine (without submachines)
public:
  MachSplit();	// create initial sequence with no machine
  virtual ~MachSplit();
  virtual MachSplit* Clone();			// create a copy of the machine and all submachines
  virtual int GetMType() {return file_header_mtype_msplit;};	// get type of machine

    // redefine connecting functions
  virtual void SetDataIn(REAL*);		// set pointer of input data
  virtual void MachAdd(Mach*); 			// add new machine after the existing ones
  virtual Mach *MachDel();

    // new functions to access individual machines
  virtual void SetGradOut(REAL *g, int mid);	// set pointer of output gradient for a particular machine
  virtual REAL* GetDataOut(int mid);		// get pointer to output data of a particular machine
 
    // this functions can't be called for a split machine since there is no unified data
  virtual REAL* GetDataOut() {			// get pointer to output data
    printf("WARNING: MachSplit::GetDataOut() has no output data for the whole machine\n");
    return NULL;
  } 
  virtual REAL* GetGradOut() {			 // return pointer on output gradient for chaining
    printf("WARNING: MachSplit::GetGradOut() has no output gradient for the whole machine\n");
    return NULL;
  }
  virtual void SetGradOut(REAL *g) {		 // set pointer of output gradient
    printf("WARNING: MachSplit::SetGradOut() has no output gradient for the whole machine\n");
  }

    // standard functions
  virtual void Info(bool=false, char *txt=(char*)"");	// display (detailed) information on machine
  virtual void Forw(int=0, bool=false);	// calculate outputs for current inputs
  virtual void Backw(const float lrate, const float wdecay, int=0);	// calculate gradients at input for current gradients at output
};

#endif
