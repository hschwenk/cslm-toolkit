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
 */

using namespace std;
#include <iostream>
#include <unistd.h>
#include <time.h>

#include "Tools.h"
#include "ErrFctSoftmCrossEntNgram.h"
#include "Blas.h"

ErrFctSoftmCrossEntNgram::ErrFctSoftmCrossEntNgram(Mach &mach)
 : ErrFct(mach)
{
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  err = Gpu::Alloc(1, "ErrFctSoftmCrossEntNgram: err variable");
#endif
}

ErrFctSoftmCrossEntNgram::ErrFctSoftmCrossEntNgram(const ErrFctSoftmCrossEntNgram &efct)
 : ErrFct(efct)
{
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  err = Gpu::Alloc(1, "ErrFctSoftmCrossEntNgram: err variable");
#endif
}

ErrFctSoftmCrossEntNgram::~ErrFctSoftmCrossEntNgram()
{
#ifdef BLAS_CUDA
  if (err) cudaFree(err);
#endif
}

//*********************************************************************************r
// E = log(sum_i d_i ln o_i)
//   = ln o_t     where t is the target index
//   output: dimension voc_size
//   target: dimension 1 with values [0,voc_size[
// We also take the log since this can't be done later if bsize>1

REAL ErrFctSoftmCrossEntNgram::CalcValue(int eff_bsize)
{
  if (eff_bsize<=0) eff_bsize=bsize;

#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  return Gpu::ErrFctSoftmCrossEntNgramCalcValue(eff_bsize, dim, output, target);
#else
  REAL *tptr=target;
  REAL *optr=output;
  double lerr=0.0;

  for (int b=0; b<eff_bsize; b++) {
    lerr += safelog(optr[(uint) *tptr++]);
    optr += dim;
  }
  return (REAL) lerr;
#endif
}

//*********************************************************************************r
// special version of CalcValue which handles NULL_WORD targets
// to be used together with CalcGradNull()

REAL ErrFctSoftmCrossEntNgram::CalcValueNull(int eff_bsize)
{
  if (eff_bsize<=0) eff_bsize=bsize;

#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  return Gpu::ErrFctSoftmCrossEntNgramCalcValueNull(eff_bsize, dim, output, target);
#else
  REAL *tptr=target;
  REAL *optr=output;
  double lerr=0.0;
  int tidx;

  for (int b=0; b<eff_bsize; b++) {
    tidx=(int) *tptr++;
    if (tidx!=NULL_WORD) lerr += safelog(optr[tidx]);
    optr += dim;
  }
  return (REAL) lerr;
#endif
}

//*********************************************************************************r
// CalcValueBatch 

void ErrFctSoftmCrossEntNgram::CalcValueBatch(int eff_bsize, REAL *res)
{
  if (eff_bsize<=0) eff_bsize=bsize;

#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  return Gpu::ErrFctSoftmCrossEntNgramCalcValueBatch(eff_bsize, dim, output, target, res);
#else
  REAL *tptr=target;
  REAL *optr=output;

  for (int b=0; b<eff_bsize; b++) {
    int tidx = (int) *tptr++;
    *res++ = (tidx==NULL_WORD) ? LOG_PROBA_NONE : safelog(optr[tidx]);
    optr += dim;
  }
#endif
}

//*********************************************************************************r
// CalcMax 

void ErrFctSoftmCrossEntNgram::CalcMax(int eff_bsize, REAL *res, int *pos)
{
  if (eff_bsize<=0) eff_bsize=bsize;

#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  return Gpu::ErrFctSoftmCrossEntNgramCalcMax(eff_bsize, dim, output, target, res, pos);
#else
  REAL *optr=output;

  for (int b=0; b<eff_bsize; b++) {
    REAL max=*optr++;
    int idx=0;
      // find max over all outputs of the current example in mini-batch
    for (int i=1; i<dim; i++,optr++) {
      if (*optr>max) { max=*optr; idx=i; }
    }
    *res++ = max;
    *pos++ = idx;
  }
#endif
}




//**********************************************************************************
// returns the target for one example in the minibatch
// idx should be in [0,bsize)
// (special version which handles NULL_WORDs)

#if 0 // not used any more use CalcValueBatch instead
REAL ErrFctSoftmCrossEntNgram::CalcValueNth(int idx)
{
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  return Gpu::ErrFctSoftmCrossEntNgramCalcValueNth(eff_bsize, dim, output, target);
#else
  REAL	*optr=output + idx*dim;	// softmax dim
  REAL	*tptr=target + idx*1;	// target dim is 1 !

  if ((int) *tptr == NULL_WORD) return -1;
  return safelog(optr[(int) *tptr]);
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

REAL ErrFctSoftmCrossEntNgram::CalcGrad(int eff_bsize)
{
  if (eff_bsize<=0) eff_bsize=bsize;

#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  Gpu::ErrFctSoftmCrossEntNgramCalcGrad(eff_bsize, dim, output, grad, target, err);
  REAL res = 0;
  Gpu::MemcpyAsync(&res, err, sizeof(REAL), cudaMemcpyDeviceToHost);
  Gpu::StreamSynchronize();
  return res;
#else

  REAL *optr=output;
  REAL *gptr=grad;
  REAL *tptr=target;
  uint	tidx;
  err=0.0;
  int n=eff_bsize*dim; REAL f1=-1.0;

  memcpy(grad,output,n*sizeof(REAL));
  SCAL(&n,&f1,grad,&inc1);
  for (int b=0; b<eff_bsize; b++) {
    if (*tptr<0.0) ErrorN("negative index %f at %d",*tptr,b);
    tidx=(uint) *tptr++;
    gptr[tidx] += 1.0;
    err += safelog(optr[tidx]);
    gptr+=dim; optr+=dim;
  }

  return err;
#endif
}


REAL ErrFctSoftmCrossEntNgram::CalcGradNull(int eff_bsize)
{
  if (eff_bsize<=0) eff_bsize=bsize;

#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  Gpu::ErrFctSoftmCrossEntNgramCalcGradNull(eff_bsize, dim, output, grad, target, err);
  REAL res = 0;
  Gpu::MemcpyAsync(&res, err, sizeof(REAL), cudaMemcpyDeviceToHost);
  Gpu::StreamSynchronize();
  return res;
#else

  REAL *optr=output;
  REAL *gptr=grad;
  REAL *tptr=target;
  int	tidx;
  err=0.0;
  int n=eff_bsize*dim; REAL f1=-1.0;

  memcpy(grad,output,n*sizeof(REAL));
  SCAL(&n,&f1,grad,&inc1);
  for (int b=0; b<eff_bsize; b++) {
    tidx=(int) *tptr++;
    if (tidx==NULL_WORD) {
      memset(gptr, 0, dim*sizeof(REAL));
    }
    else {
      gptr[tidx] += 1.0;
      err += safelog(optr[tidx]);
    }
    gptr+=dim; optr+=dim;
  }

  return err;
#endif
}

