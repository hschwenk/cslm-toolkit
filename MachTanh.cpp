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
#include "MachTanh.h"
#include "Blas.h"
#ifdef CUDA
#  include "Gpu.cuh"
#endif

MachTanh::MachTanh(const int p_idim, const int p_odim, const int p_bsize, const ulong p_nbfw, const ulong p_nbbw, const int shareid, const bool xdata)
 : MachLin(p_idim, p_odim, p_bsize, p_nbfw, p_nbbw, shareid, xdata)
{
  debug0("** constructor MachTanh\n");
#ifdef BLAS_CUDA
  tmp_tanh = Gpu::Alloc(odim*bsize, "temporary memory for tanh machine");
#endif
}

MachTanh::MachTanh(const MachTanh &m)
 : MachLin(m)
{
  debug0("** copy constructor MachTanh\n");
#ifdef BLAS_CUDA
  tmp_tanh = Gpu::Alloc(odim*bsize, "temporary memory for tanh machine");
#endif
}

MachTanh::~MachTanh()
{
  debug1("** destructor MachTanh %lx\n",(luint) this);
#ifdef BLAS_CUDA
  if (tmp_tanh) cublasFree(tmp_tanh);
#endif
}


//-----------------------------------------------
// Tools
//-----------------------------------------------

void MachTanh::Info(bool detailed, char *txt)
{
  if (detailed) {
    cout << "Information on tanh machine" << endl;
    MachLin::Info(detailed,txt);
  }
  else {
    if(Mach::fileid >= file_header_version4)
      printf("%sMachTanh %c%c[%d]-%d, bs=%d, ", txt, bExternal?'s':'p', iShareId!=-1?iShareId+'0':'-', idim, odim, bsize);
    else
      printf("%sMachTanh %d-%d, bs=%d, ", txt, idim, odim, bsize);

    if (drop_out>0) printf("drop-out=%4.2f, ", drop_out);
    printf("passes=%lu/%lu", nb_forw, nb_backw);
    if (lr_coeff != 1.0) printf(", lrate-coeff=%.2f", lr_coeff);

#ifdef BLAS_CUDA
    printf(", on GPU %d", Gpu::GetCudaDevice(Gpu::GetDevice(gpu_conf)));
#endif
    //printf(", this=%p",this);
    tm.disp(", ");
    tmh.disp(" + tanh: ");
    printf("\n");
    debug5("%s   data: %p -> %p, grad %p <- %p\n", txt, (void*)data_in, (void*)data_out, (void*)grad_in, (void*)grad_out);
  }
}

//-----------------------------------------------
// Training
//-----------------------------------------------

void MachTanh::Forw(int eff_bsize, bool in_train)
{
  debug3("*** MachTanh::Forw: mach=%p data: %p <- %p\n", this, data_in, data_out);

  if (eff_bsize<=0) eff_bsize=bsize;
  MachLin::Forw(eff_bsize,in_train);

  tmh.start();

    // apply tanh() on output
  int s=eff_bsize*odim;
#ifdef BLAS_CUDA
  Gpu::ElemwiseTanh(s, data_out, data_out); // CUDA device already set by MachLin::Forw()
#else
  VTANH(&s,data_out);
#endif

    // perform drop-out
  MachLin::ForwDropout(eff_bsize,in_train); 

  tmh.stop();
}

void MachTanh::Backw(const float lrate, const float wdecay, int eff_bsize)
{
  debug3("*** MachTanh::Backw: mach=%p grad: %p <- %p\n", this, grad_in, grad_out);
    // derivate tanh activation function
    // multiply grad_hidden by derivatives of hidden layer activities (tanh)
    // grad_out = grad_out .* f'(data_out)
    //          = grad_out .* ( 1 - data_out^2 )

  if (eff_bsize<=0) eff_bsize=bsize;
  if (!grad_out)
    Error("MachTanh::Backw(): output gradient is not set");

  tmh.start();

  int d=odim*eff_bsize;
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
# ifdef DEBUG
  { REAL buf[d];
    cublasGetVector(d,sizeof(REAL),data_out,1,buf,1);
    debug4(" output : %e %e .. %e %e\n", buf[0],buf[1],buf[d-2],buf[d-1]);
    cublasGetVector(d,sizeof(REAL),grad_out,1,buf,1);
    debug4(" grads_out: %e %e .. %e %e\n", buf[0],buf[1],buf[d-2],buf[d-1]);
  }
# endif
  // work inplace in grad_out
  Gpu::ElemwiseTanhGrad(d, data_out, grad_out, grad_out);
# ifdef DEBUG
  { REAL buf[d];
    cublasGetVector(d,sizeof(REAL),grad_out,1,buf,1);
    debug4(" grad deriv %e %e .. %e %e\n", buf[0],buf[1],buf[d-2],buf[d-1]);
  }
# endif
#else
  VSQR(&d,data_out);
  debug4(" output^2 : %e %e .. %e %e\n", data_out[0],data_out[1],data_out[d-2],data_out[d-1]);
  debug4(" grads_out: %e %e .. %e %e\n", grad_out[0],grad_out[1],grad_out[d-2],grad_out[d-1]);
  REAL *aptr = data_out;
  REAL *gptr = grad_out;
  for (int i=0; i<d; i++) *gptr++ *= (1.0 - *aptr++);	// TODO: can we use more MKL ?
  debug4(" grad deriv %e %e .. %e %e\n", grad_out[0],grad_out[1],grad_out[d-2],grad_out[d-1]);
#endif

  tmh.stop();

  MachLin::Backw(lrate, wdecay, eff_bsize);
}

