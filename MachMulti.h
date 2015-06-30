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
 * virtual class to support various combinations of multiple machines
 *
 * IMPORTANT:
 *   MultMach will call the destructor of the individual machines !
 *   Therefore the individual machines should be created by "new Mach()" and not
 *   as the address of a machine allocated as (local) variable which would be freed
 */

#ifndef _MachMulti_h
#define _MachMulti_h

using namespace std;
#include <vector>

#include "Mach.h"

class MachMulti : public Mach
{
protected:
  vector<Mach*> machs;
  vector<bool>	activ_forw;			// possibility to activate individual machines for the forward or backward pass
  vector<bool>	activ_backw;			// it's the responsibility of the user to verify that the output/gradient are still meaningful
  virtual void ReadParams(istream&, bool =true);
  virtual void ReadData(istream&, size_t, int=0);	// read binary data
  virtual void WriteParams(ostream&);		// write all params
  virtual void WriteData(ostream&);		// write binary data
  virtual void CloneSubmachs(const MachMulti &);	// copy all submachines
  MachMulti(const MachMulti &);			// create a copy of the machine (without submachines)
public:
  MachMulti();							// create initial sequence with no machine
  virtual ~MachMulti();
  virtual MachMulti* Clone();			// create a copy of the machine and all submachines
  virtual int GetMType() {return file_header_mtype_multi;};	// get type of machine
  virtual ulong GetNbParams();					// return the nbr of allocated parameters 
  virtual void SetBsize(int bs);
  virtual void SetNbEx(ulong nf, ulong nb);
    // access to machines
  virtual int MachGetNb() {return machs.size(); };		// get number of machines
  virtual Mach* MachGet(size_t);				// get pointer to a machine
    // add and remove machines
  virtual void Delete();					// call destructor for all the machines
  virtual void MachAdd(Mach*);					// add new machine after the existing ones
  virtual Mach *MachDel();					// delete the last machine
    // standard functions
  virtual void Info(bool=false, char *txt=(char*)"");		// display (detailed) information on machine
  virtual bool CopyParams(Mach*);	// copy parameters from another machine
  virtual void Forw(int=0, bool=false); 					// calculate outputs for current inputs
  virtual void Backw(const float lrate, const float wdecay, int=0); // calculate gradients at input for current gradients at output
    // additional functions
  virtual void Activate(int, bool, bool);			// selectively activate a machine for forw or backw, numbering starts at zero
};

#endif
