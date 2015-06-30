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
 * sigmoidal machine:  output = tanh(weights * input + biases)
 */

#ifndef _MachSig_h
#define _MachSig_h

#include "MachLin.h"

class MachSig : public MachLin
{
protected:
  MachSig(const MachSig &);			// create a copy of the machine
public:
  MachSig(const int=0, const int=0, const int=128, const ulong=0, const ulong=0, const int shareid=-1, const bool xdata=false);	
  virtual ~MachSig();
  virtual MachSig *Clone() {return new MachSig(*this);}	// create a copy of the machine
  virtual int GetMType() {return file_header_mtype_sig;};	// get type of machine
  virtual void Info(bool=false, char *txt= (char*)"");	// display (detailed) information on machine
  virtual void Forw(int=0, bool=false);	// calculate outputs for current inputs
    // backprop gradients from output to input and update all weights
  virtual void Backw (const float lrate, const float wdecay, int=0);
};

#endif
