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

using namespace std;
#include <iostream>
#include <unistd.h>
#include <time.h>

#include "Tools.h"
#include "ErrFctSoftmCrossEntNgramMulti.h"
#include "Blas.h"

#ifdef BLAS_CUDA
#include "Gpu.cuh"
#endif

ErrFctSoftmCrossEntNgramMulti::ErrFctSoftmCrossEntNgramMulti(Mach &mach, int n)
 : ErrFctSoftmCrossEntNgram(mach), nb(n) // this allocates memory before we change the variable "dim"
{
  if (mach.GetOdim()%nb != 0)
    Error("ErrFctSoftmCrossEntNgramMulti: output layer size is not an integer multiple");
  dim = mach.GetOdim() / nb;
  debug2("ErrFctSoftmCrossEntNgramMulti: %d n-grams of size %d\n", nb, dim);
}


//**********************************************************************************
// E = log(sum_i d_i ln o_i)
//   = ln o_t     where t is the target index
//   output: dimension voc_size
//   target: dimension 1 with values [0,voc_size[
// We also take the log since this can't be done later if bsize>1

REAL ErrFctSoftmCrossEntNgramMulti::CalcValue(int eff_bsize)
{
  if (eff_bsize<=0) eff_bsize=bsize;

#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  Error("TODO");
  return 0;
  //return Gpu::ErrFctSoftmCrossEntNgramMultiCalcValue(eff_bsize, dim, nb, output, target); 
#else
  REAL *tptr=target;
  REAL *optr=output;
  double err=0.0;

  for (int b=0; b<eff_bsize; b++) {
    for (int n=0; n<nb; n++) {
      int tidx=(int) *tptr++;
      if (tidx==NULL_WORD) {
        debug3("b=%d, n=%d, tidx=NULL, out=%f\n", b, n, optr[tidx]);
      }
      else {
        debug4("b=%d, n=%d, tidx=%d, out=%f\n", b, n, tidx, optr[tidx]);
        err += log(optr[tidx]);
      }
      optr += dim;
    }
    debug1("err=%f\n",err);
  }
  return (REAL) err;
#endif
}


//*********************************************************************************r
// CalcValuBatch 

void ErrFctSoftmCrossEntNgramMulti::CalcValueBatch(int eff_bsize, REAL *res)
{
  if (eff_bsize<=0) eff_bsize=bsize;

#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  Error("TODO: ErrFctSoftmCrossEntNgramMulti::CalcValueBatch() for GPU");
  //Gpu::ErrFctSoftmCrossEntNgramMultiCalcValueBatch(eff_bsize, dim, output, target, res);
#else
  REAL *tptr=target;
  REAL *optr=output;

  for (int b=0; b<eff_bsize; b++) {
    for (int n=0; n<nb; n++) {
      int tidx=(int) *tptr++;
      *res++ = (tidx==NULL_WORD) ? LOG_PROBA_NONE : safelog(optr[tidx]);
      optr += dim;
    }
  }
#endif
}

//**********************************************************************************
// E = log(sum_i d_i ln o_i)
//   = ln o_t     where t is the target index
//   output: dimension voc_size
//   target: dimension 1 with values [0,voc_size[
// We also take the log since this can't be done later if bsize>1

#if 0 // not used any more use CalcValueBatch instead
REAL ErrFctSoftmCrossEntNgramMulti::CalcValueNth(int idx)
{
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  Error("CUDA:  ErrFctSoftmCrossEntNgramMulti::CalcValueNth() not implemented");
  return 0.0;
#else
  Error("ErrFctSoftmCrossEntNgramMulti::CalcValueNth() not yet implemented");
  REAL	*optr=output + idx*nb*dim;
  REAL	*tptr=target + idx*nb;

  double err=0.0;
  for (int d=0; d<dim; d++)
      err += log(optr[(int) *tptr++]);
      optr += dim;
    
  return log(optr[(int) *tptr]);
#endif
}
#endif


// We include here the derivation of the softmax outputs since we have
//   dE/da_k = sum_i dE/do_i do_i/da_k
// Due to the sum, dE/do_i and do_i/da_k can't be calculated separately
// dE/do_i = d_i/o_i
// do_i/da_k = o_i (kronecker_ik - o_k)
//  -> dE/da_k = sum_i d_i/o_i * o_i (kronecker_ik - o_k)
//             = sum_i d_i (kronecker_ik - o_k)
//             = (kronecker_tk - o_k)       since d_i=0 for i!=t
REAL ErrFctSoftmCrossEntNgramMulti::CalcGrad(int eff_bsize)
{
  if (eff_bsize<=0) eff_bsize=bsize;
  debug3("ErrFctSoftmCrossEntNgramMulti::CalcGrad(): %d layers of dim %d, bsize=%d\n", nb, dim, eff_bsize);

#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  REAL err = Gpu::ErrFctSoftmCrossEntNgramMultiCalcGrad(eff_bsize, dim, nb, output, grad, target);
  debug1("ErrFctSoftmCrossEntNgramMulti::CalcGrad err=%f CUDA\n", err);
  return err;
#else

  REAL *optr=output;
  REAL *gptr=grad;
  REAL *tptr=target;
  int tidx;
  REAL err=0.0;
  int n=eff_bsize*nb*dim;
  REAL f1=-1.0;

  memcpy(grad,output,n*sizeof(REAL));
  SCAL(&n,&f1,grad,&inc1);		// TODO: can be speed-up since many phrase are incomplete
  for (int b=0; b<eff_bsize; b++) {
    for (n=0; n<nb; n++) {
      tidx=(int) *tptr++;
      if (tidx==NULL_WORD) {
        debug4("grad ngram-multi:  b=%d, n=%d, tidx=NULL, out=%f -> err=%e\n", b, n, optr[tidx], err);
	memset(gptr, 0, dim*sizeof(REAL));
      }
      else {
        gptr[tidx] += 1.0;
        err += log(optr[tidx]);
        debug6("grad ngram-multi:  b=%d, n=%d, tidx=%u, out=%f -> err=%e, grad@target=%e\n", b, n, tidx, optr[tidx], err, gptr[tidx]);
      }
      gptr+=dim; optr+=dim;
    }
  }
  debug1("ErrFctSoftmCrossEntNgramMulti::CalcGrad err=%f\n", err);

  return err;
#endif
}


