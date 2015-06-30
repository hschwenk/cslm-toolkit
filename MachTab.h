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
 * Table lookup machine:
 *   - input = index in table
 *   - output = ith line of table
 */

#ifndef _MachTab_h
#define _MachTab_h

#include <pthread.h>
#include <vector>
#include "Mach.h"
#include "Shareable.h"

class MachTab : public Mach, public Shareable
{
private:
#ifdef BLAS_CUDA
  REAL *tmp_inp;	// temporary storage to get machine back to host
#endif
  virtual void do_alloc();	// perform allocation of dynamic data structures
protected:
  REAL *t;		// look-up table
  int *t_shared;	// number of objects sharing the look-up table //Loic: why pointer and not int?
  pthread_mutex_t *t_mutex;	// mutex used to share look-up table
  virtual void WriteParams(ostream&);
  virtual void ReadParams(istream&, bool =true);
  virtual void ReadData(istream&, size_t, int=0); // read binary data
  virtual void WriteData(ostream&); // write binary data
  virtual int GetIdim() {return 1; } // we use idim internally as the dim of the table entries
  MachTab(const MachTab &);			// create a copy of the machine
public:
  MachTab(const int=1, const int=1, const int=128, const ulong=0, const ulong=0, const int=-1, const bool=false); // TODO: idim,odim init ??
  virtual ~MachTab();
  virtual MachTab *Clone() {return new MachTab(*this);}	// create a copy of the machine
  virtual int GetMType() {return file_header_mtype_tab;};	// get type of machine
  virtual ulong GetNbParams() {return bExternal ? 0 : idim*odim;}	// return the nbr of allocated parameters 
  virtual int GetMaxInpVal() {return idim;}		// all input values must be smaller than this, if not segfault ...
  virtual void TableConst(const REAL val);		// init table with constant values
  virtual void TableRandom(const REAL range);		// random init of table in [-range, range]
  virtual REAL *GetTabAdr() {return t; }		// 
  virtual void SetTabAdr(REAL *p_adr) {t=p_adr; }	// 
  virtual void FreeTabAdr() { if(t) delete t; t=NULL; }	// 
  virtual void Info(bool=false, char *txt=(char*)"");	// display (detailed) information on machine
  virtual bool CopyParams(Mach*);	// copy parameters from another machine
  virtual void Forw(int=0, bool=false);	// calculate outputs for current inputs
    // backprop gradients from output to input and update all weights
  virtual void Backw (const float lrate, const float wdecay, int=0);
  REAL *WeightTable(int &idm, int &odm);
};

#endif
