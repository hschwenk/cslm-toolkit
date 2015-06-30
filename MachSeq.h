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

#ifndef _MachSeq_h
#define _MachSeq_h

using namespace std;
#include <vector>

#include "MachMulti.h"

class MachSeq : public MachMulti
{
private:
  Timer tbackw;
protected:
  virtual void ReadData(istream&, size_t, int=0); 	// read binary data
  MachSeq(const MachSeq &);			// create a copy of the machine (without submachines)
public:
  MachSeq();	// create initial sequence with no machine
  virtual ~MachSeq();
  virtual MachSeq *Clone();			// create a copy of the machine and all submachines
  virtual int GetMType() {return file_header_mtype_mseq;};	// get type of machine
    // redfine connecting functions
  virtual void SetDataIn(REAL*);	// set pointer of input data
  virtual void SetGradOut(REAL*);	// set pointer of output gradient 
    // add and remove machines
  virtual void MachAdd(Mach*); // add new machine after the existing ones
  virtual Mach *MachDel();
  virtual void MachInsert(Mach*,size_t); // insert a new machine at a specified position
    // standard functions
  virtual void Info(bool=false, char *txt=(char*)"");	// display (detailed) information on machine
  virtual void Forw(int=0, bool=false);	// calculate outputs for current inputs
  virtual void Backw(const float lrate, const float wdecay, int=0);	// calculate gradients at input for current gradients at output
};

#endif
