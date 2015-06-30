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

#ifndef _TrainerNgramClass_h
#define _TrainerNgramClass_h

#include "TrainerNgram.h"
#include "ErrFctSoftmClassCrossEntNgram.h"


class TrainerNgramClass : public TrainerNgram
{
private:
  // Local copy of the number of word in each word class in wlist
  std::vector<int> class_sizes;
protected:
  // WordList copied from Data
  WordList* wlist;
  // Pointer to errfct cast to the right derived type
  ErrFctSoftmClassCrossEntNgram* cerrfct;
  // Pointer to the output layer of mach, which has to be a MachSoftmaxClass
  MachSoftmaxClass* machclass;

  int n_classes;
  // Buffers for additional targets (the word class)
  REAL *buf_class_target;
  // (bsize x 2) buffer containing information on the word classes:
  // - The index (in the vocabulary) of the first word of that class.
  //   It acts as an offset in the output, to locate the words of that class.
  // - The number of words in that class.
  int *buf_class_target_info;
#ifdef BLAS_CUDA
  REAL *gpu_class_target;
  int *gpu_class_target_info;
#endif

public:
  TrainerNgramClass(Mach* mach, Lrate* lrate, ErrFct* errfct,
                    const char* train, const char* dev,
                    REAL wdecay=0, int max_epochs=10, int curr_epoch=0);
  TrainerNgramClass(Mach* mach, ErrFct* errfct, Data* data);
  ~TrainerNgramClass();
  virtual REAL Train();
  virtual REAL TestDev(char* fname=NULL);
};

#endif
