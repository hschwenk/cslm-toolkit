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
 * Class definition of a general error function
 */

#ifndef _ErrFct_h
#define _ErrFct_h

#include <iostream>
#include "Tools.h"
#include "Mach.h"
#include "Data.h"

#define LOG_PROBA_NONE 999	// special value of log proba to indicate that no calculation was done
				// This happens e.g. for NULL_WORD

class ErrFct
{
private:
protected:
  int	dim;			// output dimension of machine
  int	bsize;	
  REAL  *output;		// pointer to output data (stored in machine)
  REAL  *target;		// pointer to target data (stored in trainer)
  REAL  *grad;			// calculated gradient (stored in this class)
#ifdef BLAS_CUDA
  size_t	gpu_conf;		// GPU configuration index; this is needed to run on multiple devices in parallel
#endif
public:
  ErrFct(Mach&);
  ErrFct(const ErrFct&);	// we must redefine the copy constructor
#ifdef BLAS_CUDA
  virtual ~ErrFct() { cublasFree(grad); }
#else
  virtual ~ErrFct() { delete [] grad; }
#endif
  void SetOutput(REAL *p_output) {output=p_output; }
  void SetTarget(REAL *p_target) {target=p_target; }
  REAL *GetGrad() {return grad; };
#ifdef BLAS_CUDA
  size_t GetGpuConfig() { return gpu_conf; }  // return GPU configuration index used
#endif
  virtual REAL CalcValue(int=0);		// Calculate value of error function = sum over all examples in minibatch
  virtual REAL CalcValueNull(int=0) {		//   special version that checks for NULL targets
    Error("ErrFct::CalcValueNull() should be overriden\n"); return 0.0;
  }
  virtual void CalcValueBatch(int, REAL*);	// Calculate value of error function, returns array for all values in mini batch
						//   (the vector must be allocated by the caller)	
  virtual void CalcMax(int, REAL*, int*);	// returns max value (over all outputs) and index for each example in minibatch
						//   (the vectors must be allocated by the caller)	
  virtual REAL CalcGrad(int=0);			// calculate NEGATIF gradient of error function
  virtual REAL CalcGradNull(int=0) {		//   special version that checks for NULL targets
    Error("ErrFct::CalcGradNull() should be overriden\n");
    return 0.0;
  }
#ifdef BLAS_CUDA
  virtual void CalcGradCumul(int eff_bsize) { Error("override ErrFct::CalcGradCumul()\n"); }
  virtual void InitGradCumul() { Error("override ErrFct::SetGradCumul()\n"); }
  virtual REAL GetGradCumul() { Error("override ErrFct::GetGradCumul()\n"); return 0; }
#endif
};

#endif
