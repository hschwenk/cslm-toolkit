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
 */

#ifndef _Trainer_h
#define _Trainer_h

#include <iostream>
#include "Tools.h"
#include "Mach.h"
#include "Lrate.h"
#include "ErrFct.h"
#include "Data.h"


class Trainer
{
private:
protected:
  Mach  *mach;			// network to train
  Lrate *lrate;			// learning rate to use
  ErrFct *errfct;		// error function to use
  Data  *data_train;		// training data to use
  Data  *data_dev;		// development data to use
  bool  data_dev_alloc;		//   was it allocated by this class ?
    // buffer to store bsize examples
  REAL  *buf_input;
  REAL  *buf_target;
#ifdef BLAS_CUDA		// we need to allocate this on the GPU card for data exchange
  REAL  *gpu_input;		// copied from trainer to GPU
  REAL  *gpu_target;		// copied from trainer to GPU
  REAL  *host_output;		// copied from GPU to host (needed to extract results)
#endif
    // weight decay
  REAL wdecay;
    // stats
  int nb_ex;			// during one epoch
  int nb_epoch;			// total nb of epochs
  int max_epoch;		// max numebr of epochs
  int idim, odim, bsize;	// copied here for faster access
  int auxdim, iaux;		// dimension and index of auxiliary data
  REAL err_train;		// average error during training
  REAL err_dev;			// average error during testing
  Timer tg;			// measure time to calculate the gradient
   // internal helper functions
  virtual bool Converged();	// return TRUE if training has converged or should be stopped
  virtual void StartMessage();	// dump intial message before starting training
  virtual void InfoPre();	// dump information before starting a new training epoch
  virtual void InfoPost();	// dump information after finishing a training epoch
public:
  Trainer(Mach*, Lrate*, ErrFct*,	// mach, lrate, errfct
	  const char*, const char*,		// train, dev
	  REAL =0, int =10, int =0, bool =true);	// wdecay, max epochs, current epoch, alloc target
  virtual ~Trainer();
  virtual REAL Train();				// train for one epoch
  virtual REAL TestDev(char* =NULL);		// test current network on dev data and save outputs into file
						// returns obtained error (-1 if error)
  virtual void TrainAndTest(const char*);	// main training routine for X iterations
};

#endif
