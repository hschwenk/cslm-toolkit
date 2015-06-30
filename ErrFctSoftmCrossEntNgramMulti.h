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
 * Class definition of cross entropy error function
 * Special version for NNs that predict MULTIPLE words
 *  - the NN has a large output dimension (vocsize or limited to shortlist)
 *  - the data has one dimensional targets that are taken as index into
 *  	the word list
 *  - therefore the target vector is binary: 1 at the position of the to predicted
 *  	word, 0 elsewhere
 *
 *   E = sum_i  d_i * ln o_i
 *   dE/do_k = d_k / o_k   for o_k <> 0
 * This is usually used with softmax outputs
 */

#ifndef _ErrFctSoftmCrossEntNgramMulti_h
#define _ErrFctSoftmCrossEntNgramMulti_h

#include <iostream>
#include "Tools.h"
#include "ErrFct.h"
#include "ErrFctSoftmCrossEntNgram.h"
#ifdef BLAS_CUDA
#  include "Gpu.cuh"
#endif


class ErrFctSoftmCrossEntNgramMulti : public ErrFctSoftmCrossEntNgram
{
private:
  int nb;		// number of separate output n-gram each of dimension "dim"
			// -> therefore the total size of the gradient is nb*dim !!
public:
  ErrFctSoftmCrossEntNgramMulti(Mach &mach, int n);
  virtual REAL CalcValue(int=0);		// Calculate value of error function = sum over all examples in minibatch
  virtual void CalcValueBatch(int, REAL*);	// Calculate value of error function, returns array for all values in mini batch
						//   (the vector must be allocated by the caller)	
  virtual void CalcMax(int, REAL*, int*) { printf ("ERROR: Unimplemenetd function");};	// returns max value (over all outputs) and index for each example in minibatch
						//   (the vectors must be allocated by the caller)	
  virtual REAL CalcGrad(int=0);		// calculate NEGATIF gradient of error function
#ifdef BLAS_CUDA
  virtual void InitGradCumul() { Gpu::ResSet(0.0); };
  virtual REAL GetGradCumul() { return Gpu::ResGet(); };
#endif
};

#endif
