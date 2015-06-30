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
 *  Parallel machine:
 *   - put several machine in parallel with a concatenated input and output layer
 *   - the dimensions of the input and output layers may be different
 *
 *  Memory allocation:
 *   - we need to copy and re-arrange all input and ouput data
 *     since each individual machines expects a batch of its data in continuous memory allocations
 *
 *     This was buggy until January 2015, but had no effect if only MachTab with shared weights are used
 *
 */

#ifndef _MachPar_h
#define _MachPar_h

using namespace std;
#include <vector>

#include "MachMulti.h"

class MachPar : public MachMulti
{
private:
  void do_alloc();	// perform allocation of dynamic data structures
protected:
  virtual void ReadData(istream&, size_t, int=0); // read binary data
  MachPar(const MachPar &);			// create a copy of the machine (without submachines)
public:
  MachPar();	// create initial sequence with no machine
  virtual ~MachPar();
  virtual MachPar *Clone();			// create a copy of the machine and all submachines
  virtual int GetMType() {return file_header_mtype_mpar;};	// get type of machine
    // add and remove machines
  virtual void MachAdd(Mach*); // add new machine after the existing ones
  virtual Mach *MachDel();
    // standard functions
  virtual void Info(bool=false, char *txt=(char*)"");	// display (detailed) information on machine
  virtual void Forw(int=0, bool=false);	// calculate outputs for current inputs
  virtual void Backw(const float lrate, const float wdecay, int=0);	// calculate gradients at input for current gradients at output
};

#endif
