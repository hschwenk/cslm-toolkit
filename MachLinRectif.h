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
 * machine with linear rectifier units:  output(x) = x if x>0, 0 else 
 */

#ifndef _MachLinRectif_h
#define _MachLinRectif_h

#include "MachLin.h"

class MachLinRectif : public MachLin
{
private:
  Timer tmh;			// cumulated time used for backprop normalization
protected:
  MachLinRectif(const MachLinRectif &);			// create a copy of the machine
public:
  MachLinRectif(const int=0, const int=0, const int=128, const ulong=0, const ulong=0, const int shareid=-1, const bool xdata=false);	
  virtual ~MachLinRectif();
  virtual MachLinRectif *Clone() {return new MachLinRectif(*this);}	// create a copy of the machine
  virtual int GetMType() {return file_header_mtype_lin_rectif;};	// get type of machine
  virtual void Info(bool=false, char *txt=(char*)"");	// display (detailed) information on machine
  virtual void Forw(int=0, bool=false);	// calculate outputs for current inputs
    // backprop gradients from output to input and update all weights
  virtual void Backw (const float lrate, const float wdecay, int=0);
};

#endif
