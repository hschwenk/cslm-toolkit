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
#include "MachSoftmaxStable.h"
#include "Blas.h"
#ifdef BLAS_CUDA
# include "Gpu.cuh"
#endif


MachSoftmaxStable::MachSoftmaxStable(const int p_idim, const int p_odim, const int p_bsize, const ulong p_nbfw, const ulong p_nbbw, const int shareid, const bool xdata)
 : MachLin(p_idim, p_odim, p_bsize, p_nbfw, p_nbbw, shareid, xdata)
{
  debug0("** constructor MachSoftmaxStable\n");
#if defined(BLAS_CUDA) && defined(BLAS_CUDA_NPPS_SUM)
  int nbytes=0;
  Gpu::SetConfig(gpu_conf);
  nppsSumGetBufferSize_32f(odim, &nbytes);
  debug2(" - CUDA MachSoftmaxStable: allocating %d bytes for fast sum of %d-dimensional output layer\n",nbytes,odim);
  gpu_sum_buf = nppsMalloc_8u(nbytes);
#endif
#ifdef BLAS_CUDA
  if(Gpu::GetDeviceProp(gpu_conf).warpSize != 32){
    Error("KernelSoftmax used by MachSoftmaxStable suppose a wrapSize of 32. The code will return wrong result if run!");
  } 
#endif
}

MachSoftmaxStable::MachSoftmaxStable(const MachSoftmaxStable &m)
 : MachLin(m)
{
  debug0("** copy constructor MachSoftmaxStable\n");
#if defined(BLAS_CUDA) && defined(BLAS_CUDA_NPPS_SUM)
  int nbytes=0;
  nppsSumGetBufferSize_32f(odim, &nbytes);
  debug2(" - CUDA MachSoftmaxStable: allocating %d bytes for fast sum of %d-dimensional output layer\n",nbytes,odim);
  gpu_sum_buf = nppsMalloc_8u(nbytes);
#endif
#ifdef BLAS_CUDA
  if(Gpu::GetDeviceProp(gpu_conf).warpSize != 32){
    Error("KernelSoftmax used by MachSoftmaxStable suppose a wrapSize of 32. The code will return wrong result if run!");
  }
#endif
}

MachSoftmaxStable::~MachSoftmaxStable()
{
  debug0("** destructor MachSoftmaxStable\n");
#if defined(BLAS_CUDA) && defined(BLAS_CUDA_NPPS_SUM)
  Gpu::SetConfig(gpu_conf);
  if (gpu_sum_buf) nppsFree(gpu_sum_buf);
#endif
}

//-----------------------------------------------
// Tools
//-----------------------------------------------

void MachSoftmaxStable::Info(bool detailed, char *txt)
{
  if (detailed) {
    cout << "Information on softmax stable machine" << endl;
    MachLin::Info(detailed);
  }
  else {
    printf("%sMachSoftmaxStable %d-%d, bs=%d, passes=%lu/%lu", txt,idim, odim, bsize, nb_forw, nb_backw);
    if (lr_coeff != 1.0) printf(", lrate-coeff=%.2f", lr_coeff);
#ifdef BLAS_CUDA
    printf(", on GPU %d", Gpu::GetCudaDevice(Gpu::GetDevice(gpu_conf)));
#endif
    tm.disp(", ");
    tmn.disp(" + norm: ");
    printf("\n");
    debug5("%s   data: %p -> %p, grad %p <- %p\n", txt, (void*)data_in, (void*)data_out, (void*)grad_in, (void*)grad_out);
  }
}

//-----------------------------------------------
// Training
//-----------------------------------------------

void MachSoftmaxStable::Forw(int eff_bsize, bool in_train)
{
  debug2("*** MachSoftmaxStable::Forw: %p -> %p\n",(void*)data_in,(void*)data_out);

  if (eff_bsize<=0) eff_bsize=bsize;
  MachLin::Forw(eff_bsize,in_train);

  tmn.start();

    // softmax stable normalization
#ifdef BLAS_CUDA
    // device already set by MachLin::Forw()
  Gpu::MachSoftmaxStableForw(eff_bsize,odim,data_out);
#else
  for (int b=0; b<eff_bsize; b++) {
    // Get the maximum value of data_out on row b
    REAL max = data_out[b*odim];
    for (int i=1; i<odim; i++) {
      REAL x = data_out[b*odim + i];
      if (x > max)
        max = x;
    }
    // Compute exp(x - max) inplace for each x in row b of data_out, and their sum
    REAL sum_exp = 0.;
    for (int i=0; i<odim; i++) {
      REAL exp_x = exp(data_out[b*odim + i] - max);
      sum_exp += exp_x;
      data_out[b*odim + i] = exp_x;
    }
    // Normalize the row, dividing all values by sum_exp
    for (int i=0; i<odim; i++) {
      data_out[b*odim + i] /= sum_exp;
    }
  }
#endif

    // perform drop-out
  MachLin::ForwDropout(eff_bsize,in_train); 

  tmn.stop();
}

void MachSoftmaxStable::Backw(const float lrate, const float wdecay, int eff_bsize)
{
  debug2("*** MachSoftmaxStable Backw %p <- %p\n",(void*)grad_in,(void*)grad_out);
    // derivate softmax activation function
    //   do_i / da_k = o_i (kronecker_ik - o_k)
    // we suppose that do_i/da_k vanishes in the error function !!
    //             = o_i (1 - o_i)

   // this can't be done here since the result depends
   // on the error function (we must derivate each output w/r
   // to ALL other outputs. This can't be stored in one vector)
   //   dE/da_i = sum_k dE/do_k do_k/da_i
   // On the other hand, many terms vanish with usual error functions

  MachLin::Backw(lrate, wdecay, eff_bsize);
}
