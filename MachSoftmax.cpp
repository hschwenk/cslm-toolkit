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
#include <math.h>

#include "Tools.h"
#include "MachSoftmax.h"
#include "Blas.h"
#ifdef BLAS_CUDA
# include "Gpu.cuh"
#endif


MachSoftmax::MachSoftmax(const int p_idim, const int p_odim, const int p_bsize, const ulong p_nbfw, const ulong p_nbbw, const int shareid, const bool xdata)
 : MachLin(p_idim, p_odim, p_bsize, p_nbfw, p_nbbw, shareid, xdata)
{
#if defined(BLAS_CUDA) && defined(BLAS_CUDA_NPPS_SUM)
  int nbytes=0;
  Gpu::SetConfig(gpu_conf);
  nppsSumGetBufferSize_32f(odim, &nbytes);
  gpu_sum_buf = nppsMalloc_8u(nbytes);
#endif
#ifdef BLAS_CUDA
  if(Gpu::GetDeviceProp(gpu_conf).warpSize != 32){
    Error("KernelSoftmax used by MachSoftmax supposes a wrapSize of 32. The code will return wrong result if run!");
  } 
#endif
}

MachSoftmax::MachSoftmax(const MachSoftmax &m)
 : MachLin(m)
{
#if defined(BLAS_CUDA) && defined(BLAS_CUDA_NPPS_SUM)
  int nbytes=0;
  nppsSumGetBufferSize_32f(odim, &nbytes);
  gpu_sum_buf = nppsMalloc_8u(nbytes);
#endif
#ifdef BLAS_CUDA
  if(Gpu::GetDeviceProp(gpu_conf).warpSize != 32){
    Error("KernelSoftmax used by MachSoftmax supposes a wrapSize of 32. The code will return wrong result if run!");
  }
#endif
}

MachSoftmax::~MachSoftmax()
{
#if defined(BLAS_CUDA) && defined(BLAS_CUDA_NPPS_SUM)
  Gpu::SetConfig(gpu_conf);
  if (gpu_sum_buf) nppsFree(gpu_sum_buf);
#endif
}

//-----------------------------------------------
// Tools
//-----------------------------------------------

void MachSoftmax::Info(bool detailed, char *txt)
{
  if (detailed) {
    cout << "Information on softmax machine" << endl;
    MachLin::Info(detailed);
  }
  else {
    printf("%sMachSoftmax %d-%d, bs=%d, passes=%lu/%lu", txt,idim, odim, bsize, nb_forw, nb_backw);
    if (lr_coeff != 1.0) printf(", lrate-coeff=%.2f", lr_coeff);
#ifdef BLAS_CUDA
    printf(", on GPU %d", Gpu::GetCudaDevice(Gpu::GetDevice(gpu_conf)));
#endif
    //printf(", this=%p",this);
    tm.disp(", ");
    tmn.disp(" + norm: ");
    printf("\n");
  }
}

//-----------------------------------------------
// Training
//-----------------------------------------------

void MachSoftmax::Forw(int eff_bsize, bool in_train)
{

  if (eff_bsize<=0) eff_bsize=bsize;
  MachLin::Forw(eff_bsize,in_train);

  tmn.start();

    // softmax normalization
#ifdef BLAS_CUDA
    // device already set by MachLin::Forw()
  Gpu::MachSoftmaxForw(eff_bsize,odim,data_out);
#else
  REAL *optr, sum;
  int b=eff_bsize*odim;
     // apply exp() on all outputs
  VEXP(&b,data_out);
  for (b=0,optr=data_out; b<eff_bsize; b++,optr+=odim) {
    sum=1.0/ASUM(&odim,optr,&inc1);  // exp(x) is always positive -> we can use the sum_i (ABS(x_i))
    SCAL(&odim,&sum,optr,&inc1);
  }
#endif

    // perform drop-out
  MachLin::ForwDropout(eff_bsize,in_train); 

  tmn.stop();
}

void MachSoftmax::Backw(const float lrate, const float wdecay, int eff_bsize)
{
    // derivate softmax activation function
    //   do_i / da_k = o_i (kronecker_ik - o_k)
    // we suppose that do_i/da_k vanishes in the error function !!
    //             = o_i (1 - o_i)

   // this can't be done here since the result depends
   // on the error function (we must derivate each output w/r
   // to ALL other outputs. This can't be stored in one vector)
   //   dE/da_i = sum_k dE/do_k do_k/da_i
   // On the other hand, many terms vanish with usual error functions

   // So here, we rely on the implementation in the error function
   // (ErrFctSoftmCrossEntNgram) to actually compute the gradient
   // wrt cross-entropy AND softmax (dE/da_i, NOT dE/do_i),
   // so here we only forward it to the gradient wrt the linear part.

  MachLin::Backw(lrate, wdecay, eff_bsize);
}

