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
#include "MachLinRectif.h"
#include "Blas.h"

#ifdef BLAS_CUDA
#include "Gpu.cuh"
#endif

MachLinRectif::MachLinRectif(const int p_idim, const int p_odim, const int p_bsize, const ulong p_nbfw, const ulong p_nbbw, const int shareid, const bool xdata)
 : MachLin(p_idim, p_odim, p_bsize, p_nbfw, p_nbbw, shareid, xdata)
{
  debug0("** constructor MachLinRectif\n");
}

MachLinRectif::MachLinRectif(const MachLinRectif &m)
 : MachLin(m)
{
  debug0("** copy constructor MachLinRectif\n");
}

MachLinRectif::~MachLinRectif()
{
  debug1("** destructor MachLinRectif %lx\n",(luint) this);
}


//-----------------------------------------------
// Tools
//-----------------------------------------------

void MachLinRectif::Info(bool detailed, char *txt)
{
  if (detailed) {
    cout << "Information on linear rectifier machine" << endl;
    MachLin::Info(detailed,txt);
  }
  else {
    if (drop_out>0)
      printf("%sMachLinRectif %d-%d, bs=%d, drop-out=%4.2f, passes=%lu/%lu", txt, idim, odim, bsize, drop_out, nb_forw, nb_backw);
    else 
      printf("%sMachLinRectif %d-%d, bs=%d, passes=%lu/%lu", txt, idim, odim, bsize, nb_forw, nb_backw);
    if (lr_coeff != 1.0) printf(", lrate-coeff=%.2f", lr_coeff);
#ifdef BLAS_CUDA
    printf(", on GPU %d", Gpu::GetCudaDevice(Gpu::GetDevice(gpu_conf)));
#endif
    tm.disp(", ");
    tmh.disp(" + recif: ");
    printf("\n");
    debug5("%s   data: %p -> %p, grad %p <- %p\n", txt, (void*)data_in, (void*)data_out, (void*)grad_in, (void*)grad_out);
  }
}

//-----------------------------------------------
// Training
//-----------------------------------------------

void MachLinRectif::Forw(int eff_bsize, bool in_train)
{
  debug2("*** MachLinRectif Forw %p -> %p\n",(void*)data_in,(void*)data_out);

  if (eff_bsize<=0) eff_bsize=bsize;
  MachLin::Forw(eff_bsize,in_train);

  tmh.start();

    // apply linear rectifier on output
#ifdef BLAS_CUDA
  Gpu::LinRectifForw(odim*eff_bsize, data_out);
#else
  REAL *ptr=data_out;
  for (int i=0; i<odim*eff_bsize; i++, ptr++) {
    if (*ptr<0) *ptr=0;
  }
#endif

    // perform drop-out
  MachLin::ForwDropout(eff_bsize,in_train); 

  tmh.stop();
}

void MachLinRectif::Backw(const float lrate, const float wdecay, int eff_bsize)
{
  debug2("*** MachLinRectif Backw %p <- %p\n",(void*)grad_in,(void*)grad_out);
    // derivate tanh activation function
    // multiply grad_hidden by derivatives of hidden layer activities (tanh)
    // grad_out = grad_out .* f'(data_out)
    //          = grad_out .* ( 1 - data_out^2 )

  if (eff_bsize<=0) eff_bsize=bsize;
  if (!grad_out)
    Error("MachLinRectif::Backw(): output gradient is not set");

  tmh.start();
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  Gpu::LinRectifBackw(odim*eff_bsize, data_out, grad_out);
#else
  REAL *dptr=data_out;
  REAL *gptr=grad_out;
  for (int i=0; i<odim*eff_bsize; i++) {
    if (*dptr++<0) *gptr++=0;  // multiply by 0
               else gptr++; // multiply by 1
  }
#endif
  tmh.stop();

  MachLin::Backw(lrate, wdecay, eff_bsize);
}

