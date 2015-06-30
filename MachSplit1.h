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
 *   - the output of the split machine concatenates the outputs of the individual machines
 *
 * Memory allocation:
 *  - output data:  allocated by the split machine (values are copied from indiv. machines)
 *  - input data:   all indiv machines share the same pointer on the input
 *  		    (not allocated by the split machine)
 *  - input gradient: cumulates gradients of indiv. machines
 *  		    allocated by the split machine
 */

#ifndef _MachSplit1_h
#define _MachSplit1_h

using namespace std;
#include <vector>

#include "MachMulti.h"

class MachSplit1 : public MachMulti
{
private:
  REAL *grad_out_split;	// internal output gradient to rearange the order of blocks
			// we have the same problem, but can arrange this when copying from the individual machines
  Timer tbackw;
protected:
  virtual void ReadData(istream&, size_t, int=0); // read binary data
  MachSplit1(const MachSplit1 &);			// create a copy of the machine (without submachines)
public:
  MachSplit1();	// create initial sequence with no machine
  virtual ~MachSplit1();
  virtual MachSplit1* Clone();			// create a copy of the machine and all submachines
  virtual int GetMType() {return file_header_mtype_msplit1;};	// get type of machine
    // redfine connecting functions
  virtual void SetDataIn(REAL*);	// set pointer of input data
  virtual void SetGradOut(REAL *g) {grad_out=g;}   // set pointer of output gradient (we keep an internal buf for the indiv. machs !)
    // add and remove machines
  virtual void MachAdd(Mach*); // add new machine after the existing ones
  virtual Mach *MachDel();
    // standard functions
  virtual void Info(bool=false, char *txt=(char*)"");	// display (detailed) information on machine
  virtual void Forw(int=0, bool=false);	// calculate outputs for current inputs
  virtual void Backw(const float lrate, const float wdecay, int=0);	// calculate gradients at input for current gradients at output
};

#endif
