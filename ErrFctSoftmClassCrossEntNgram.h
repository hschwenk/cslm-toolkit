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
 * Cross-entropy error function when the probability of a word is factored
 * into probability of a category times probability of the word given the category.
 * The category predictor is a softmax, and so is each of the word predictors.
 *
 * This class should be used in conjunction with MachSoftmaxClass.
 */

#ifndef _ErrFctSoftmClassCrossEntNgram_h
#define _ErrFctSoftmClassCrossEntNgram_h

#include "ErrFct.h"
#include "MachSoftmaxClass.h"


class ErrFctSoftmClassCrossEntNgram : public ErrFct
{
private:
  // Buffer for the class index of target words
  REAL* target_class;
  // Buffer for cached information about the target class: offset (wrt the full output)
  // and number of words in that class. This buffer is always in host memory,
  // as we need to perform pointer arithmetic on it.
  int* target_class_info;
  // Buffer for the predicted class probabilities
  REAL* output_class;
  // Buffer for the gradient wrt these probabilities
  REAL* grad_class;

#ifdef BLAS_CUDA
  REAL* err;
  REAL* host_err;
#endif

protected:
  int n_classes;
  std::vector<int> class_sizes;
  WordList* wlist;
  MachSoftmaxClass* mach_class;

public:
  ErrFctSoftmClassCrossEntNgram(Mach &mach);
  ErrFctSoftmClassCrossEntNgram(const ErrFctSoftmClassCrossEntNgram&);
  virtual ~ErrFctSoftmClassCrossEntNgram();
  virtual void SetUp(MachSoftmaxClass* mach_class, WordList* wlist);

  virtual void SetOutputClass(REAL* p_output_class)
  {
    output_class = p_output_class;
  }
  virtual void SetTargetClassInfo(REAL* p_target_class, int* p_target_class_info)
  {
    target_class = p_target_class;
    target_class_info = p_target_class_info;
    // The MachSoftmaxClass needs to know where the target class is,
    // so it can compute only the conditional probabilities of words
    // in that class. It does not actually need the index, just the info.
    if (mach_class) {
      mach_class->SetTargetInfo(p_target_class_info);
    }
  }
  virtual REAL* GetGradClass()
  {
    return grad_class;
  }

  virtual REAL CalcValue(int=0);		// Calculate value of error function = sum over all examples in minibatch
  virtual void CalcValueBatch(int, REAL*);	// Calculate value of error function, returns array for all values in mini batch
						//   (the vector must be allocated by the caller)	
  virtual void CalcMax(int, REAL*, int*);	// returns max value (over all outputs) and index for each example in minibatch
						//   (the vectors must be allocated by the caller)	
						
  // calculate NEGATIVE gradient of error function
  virtual REAL CalcGrad(int eff_bsize=0);
  // special version that checks for NULL targets
  virtual REAL CalcGradNull(int eff_bsize=0);

  // Compute classification error on word classes
  virtual REAL CalcWordClassError(int eff_bsize=0);

};

#endif
