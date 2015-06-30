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

#include "Tools.h"
#include "MachCopy.h"
#ifdef CUDA
#  include "Gpu.cuh"
#endif

MachCopy::MachCopy(const int p_idim, const int p_odim, const int p_bsize, const ulong p_nbfw, const ulong p_nbbw)
 : Mach(p_idim, p_odim, p_bsize, p_nbfw, p_nbbw)
{
#ifdef BLAS_CUDA
#else
 
 if (odim != idim) {
    Error ("The input size should be equal the output size for copy machine");
  }
#endif
}

MachCopy::MachCopy(const MachCopy &m)
 : Mach(m)
{
}

/*******************************************
 *
 ********************************************/

void MachCopy::Info(bool detailed, char *txt)
{
  if (detailed) {
    cout << "Information on copy machine" << endl;
    Mach::Info(detailed,txt);
  }
  else {
    printf("%sMachCopy %d-%d, bs=%d, passes=%lu/%lu", txt, idim, odim, bsize, nb_forw, nb_backw);
#ifdef BLAS_CUDA
    printf(", on GPU %d", Gpu::GetCudaDevice(Gpu::GetDevice(gpu_conf)));
#endif
    tm.disp(", ");
    tm.newline();

#ifdef BLAS_CUDA
#else
#endif
  }
}

//-----------------------------------------------
// File input
//-----------------------------------------------


void MachCopy::ReadData(istream &inpf, size_t s, int bs)
{
  if (0 != s)
    ErrorN("data block of copy machine has %zu elements (0 were expected)", s);
  Mach::ReadData(inpf, 0, bs);
}


//-----------------------------------------------
// Training
//-----------------------------------------------

void MachCopy::Forw(int eff_bsize, bool in_train)
{

  tm.start();
  
  if (!data_in)
    Error("MachCopy::Forw(): input data is not set");
  if (eff_bsize<=0) eff_bsize=bsize;

  debugMachInp("MachCopy",data_in,idim,odim,eff_bsize);

#ifdef BLAS_CUDA
    Gpu::MemcpyAsync(data_out, data_in, eff_bsize * odim * sizeof(REAL), cudaMemcpyDeviceToDevice);
#else
    memcpy(data_out, data_in, eff_bsize * odim * sizeof(REAL));
#endif
  nb_forw += eff_bsize;

  tm.stop();
  debugMachOutp("MachCopy",data_out,idim,odim,eff_bsize);
}


void MachCopy::Backw(const float lrate, const float wdecay, int eff_bsize)
{
  if (eff_bsize<=0) eff_bsize=bsize;
  if (!grad_out)
    Error("MachCopy::Backw(): output gradient is not set");

  debugMachOutp("MachCopy Grad",grad_out,idim,odim,eff_bsize);
  tm.start();
#ifdef BLAS_CUDA
    Gpu::MemcpyAsync(grad_in, grad_out, eff_bsize * odim * sizeof(REAL), cudaMemcpyDeviceToDevice);
#else
    memcpy(grad_in, grad_out, eff_bsize * odim * sizeof(REAL));
#endif

  nb_backw += eff_bsize;

  tm.stop();
  debugMachInp("MachCopy Grad",grad_in,idim,odim,eff_bsize);
}
