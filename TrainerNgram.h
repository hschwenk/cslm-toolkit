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

#ifndef _TrainerNgram_h
#define _TrainerNgram_h

#include <ostream>
#include "Tools.h"
#include "Mach.h"
#include "ErrFct.h"
#include "Data.h"
#include "DataNgramBin.h"
#include "Trainer.h"


class TrainerNgram : public Trainer
{
private:
    // copies of important fields
  int order;			// from Data
protected:
   // internal helper functions
  virtual void InfoPost();	// dump information after finishing a training epoch
  WordID  *buf_target_wid;	// used instead of buf_target to casts between REAL and WordID
public:
  TrainerNgram(Mach*, Lrate*, ErrFct*,	// mach, lrate, errfct
	  const char*, const char*,			// train, dev
	  REAL =0, int =10, int =0, bool =false);	// wdecay, max epochs, current epoch, use word class
  TrainerNgram(Mach*, ErrFct*, Data*, int=0);	// for testing only
  ~TrainerNgram();
  virtual REAL Train();					// train for one epoch
  virtual REAL TestDev(char* =NULL);			// test current network on dev data and save outputs into file
							// returns obtained error (-1 if error)
};

#endif
