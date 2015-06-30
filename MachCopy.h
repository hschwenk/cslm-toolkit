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
 * copy machine:  output = input
 */

#ifndef _MachCopy_h
#define _MachCopy_h

#include "Mach.h"

class MachCopy : public Mach
{
protected:
  virtual void ReadData(istream&, size_t, int=0); // read binary data
  MachCopy(const MachCopy &);			// create a copy of the machine, sharing the parameters
public:
  MachCopy(const int=0, const int=0, const int=128, const ulong=0, const ulong=0);	
  virtual ~MachCopy() {}
  virtual MachCopy *Clone() {return new MachCopy(*this);}		// create a copy of the machine, sharing the parameters
  virtual int GetMType() {return file_header_mtype_copy;};	// get type of machine
  virtual void Info(bool=false, char *txt=(char*)"");		// display (detailed) information on machine
  virtual void Forw(int=0, bool=false);	// calculate outputs for current inputs
    // backprop gradients from output to input and update all weights
  virtual void Backw (const float lrate, const float wdecay, int=0);
};

#endif
