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

#ifndef _Machine_h
#define _Machine_h

#include <iostream>
#include <fstream>
#include "Tools.h"
#include "Blas.h"
#include "Timer.h"

// list of all known machine types,
// this is needed for the general file read function

#define file_header_name "HPerf"
#define file_header_version1 1		// initial version
#define file_header_version2 2		// 2013/12/08: switched to ulong for nb_forw and nb_backw
#define file_header_version3 3		// 2015/03/18: added sharing information for MachTab 
#define file_header_version4 4		// 2015/06/05: generalized sharing information: all simple machine can share its weights 
#define file_header_version file_header_version4
#define file_header_size 16

#define file_header_mtype_base		0
#define file_header_mtype_tab		1
#define file_header_mtype_tabsh		2
#define file_header_mtype_lin		3
#define file_header_mtype_sig		4
#define file_header_mtype_tanh		5
#define file_header_mtype_softmax	6
#define file_header_mtype_stab		7
#define file_header_mtype_softmax_stable 8
#define file_header_mtype_lin_rectif	9
#define file_header_mtype_softmax_class	10
#define file_header_mtype_copy          11
#define file_header_mtype_multi		16
#define file_header_mtype_mseq		17
#define file_header_mtype_msplit1	18
#define file_header_mtype_mpar		19
#define file_header_mtype_msplit	20
#define file_header_mtype_mjoin		21
#define file_header_mtype_combined	31
#define file_header_mtype_max		32
#define file_header_mtype_avr		33

extern int shareOffs;
class Mach
{
private:
  void do_alloc();	// perform allocation of dynamic data structures
protected:
  static int fileid;
  int idim, odim;		// input and output dimension
  int bsize;			// block size (nb of example used in parallel)
  ulong nb_forw;		// nb of forward examples processed
  ulong nb_backw;		// nb of backward examples processed
  bool update;			// update internal parameters
  REAL lr_coeff;		// machine specific learning coefficient (default 1.0)
    // drop-out
  REAL drop_out;		// dropout probability, 0: not used (default), >0 apply drop-out (in training), <0 scale weighted sum (in testing)
  REAL *drop_out_rand;		// random values for the whole output vector

#if 0
   // recurrent conncetions: user specified parameters
  uint rec_hist;		// nb of examples which are memorized
  uint rec_step;		// number of step before we do an update of the weights
  uint rec_span;		// number of step we go back during update
				// rec_span can be larger than rec_step !
				// both must be smaller or equal than rec_hist
   // recurrent conncetions: for internal handling
   // all the buffers are circular, no data is moved once stored
  uint rec_ipos;		// position in array where to add the new examples
				// starts with 0, and wraps around once the end is reached
  uint rec_len;			// numnber of examples memorized in the buffers
#endif

   // CUDA: the following four variables refer to device memory
   // the size of these buffers is: DIM * bsize * rec_hist
  REAL *data_in;		// input data (pointer)
				// CUDA: we need to allocate device memory
  REAL *data_out;		// output data (allocated by machine)
  REAL *grad_in;		// input gradients (allocated by machine)
  REAL *grad_out;		// output gradients (pointer)
				// CUDA: we need to allocate device memory

  Timer   tm;			// count real and user time
#ifdef BLAS_CUDA
  size_t	gpu_conf;  		// GPU configuration index; this is needed to run on multiple devices in parallel
#endif
  // File or stream I/O, the following functions can be overloaded by subclass
  // the main functions Read() and Write() should not be modified !
  virtual void ReadParams(istream&, bool=true); // read all params
  virtual void ReadData(istream&, size_t, int=0); // read binary data
  virtual void WriteParams(ostream&); // write all params
  virtual void WriteData(ostream&); // write binary data
  Mach(const Mach &, const int=0);			// create a copy of the machine
public:
  Mach(const int=0, const int=0, const int=128, const ulong=0, const ulong=0);
  virtual ~Mach();
  virtual Mach *Clone() {return new Mach(*this);}	// create a copy of the machine
    // Tools
  virtual int GetMType() {return file_header_mtype_base;};	// get type of machine
  virtual int GetIdim() {return idim;}
  int GetOdim() {return odim;}
  int GetBsize() {return bsize;}

  virtual void SetBsize(int bs) {
    if (bs<1) Error("wrong value in SetBsize()"); else bsize=bs; }
  ulong GetNbForw() {return nb_forw;}
  ulong GetNbBackw() {return nb_backw;}
  virtual void SetNbEx(ulong nf, ulong nb) {nb_forw=nf; nb_backw=nb;}
  virtual ulong GetNbParams() {return 0;}	// return the nbr of allocated parameters
  void SetUpdataParams(bool up) {update=up;}	// change flag to update internal parameters
  void SetLrateCoeff(REAL v) {lr_coeff=v;}
  virtual REAL* GetDataIn() {return data_in;}	// return pointer on input data for chaining
  virtual REAL* GetDataOut() {return data_out;}	// return pointer on output data for chaining
  virtual REAL* GetGradIn() {return grad_in;}	// return pointer on input gradient for chaining
  virtual REAL* GetGradOut() {return grad_out;} // return pointer on output gradient for chaining
  virtual void SetDataIn(REAL *data) {data_in=data;} // set pointer of input data
  virtual void SetGradOut(REAL *data) {grad_out=data;} // set pointer of output gradient 
  virtual void SetDropOut(const REAL v); 	// set drop-out fraction
#ifdef BLAS_CUDA
  size_t GetGpuConfig() { return gpu_conf; }	// return GPU configuration index used for this machine
#endif
  virtual void Info(bool=false, char *txt=(char*)" - ");// display (detailed) information on machine
  virtual bool CopyParams(Mach*);	// copy parameters from another machine
    // FILE IO
  static Mach *Read(istream&, int=0);	// read class from a stream
  void Write(ostream&); // write content of class to a stream
    // Training
  virtual void Forw(int=0, bool=false);	// calculate outputs for current inputs
    // backprop gradients from output to input and update all weights
  virtual void Backw (const float lrate, const float wdecay, int =0);

  static int GetFileId(){ return fileid;}
  static void SetFileId(int id){ fileid = id;}
  static bool canShare(int mtype) {
   return (mtype != file_header_mtype_base 
        && mtype != file_header_mtype_stab
        && mtype <= file_header_mtype_softmax_class);
  }
  static void SetShareOffs(int val) { shareOffs = val; }
};

void GpuUnlock();

// Find sub-machines matching desired mtype in parent_mach (depth-first).
Mach* FindFirstMatching(int mtype, Mach* parent_mach);
std::vector<Mach*> FindAllMatching(int mtype, Mach* parent_mach);

#endif
