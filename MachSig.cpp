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
#include "MachSig.h"

MachSig::MachSig(const int p_idim, const int p_odim, const int p_bsize, const ulong p_nbfw, const ulong p_nbbw, const int shareid, const bool xdata)
 : MachLin(p_idim, p_odim, p_bsize, p_nbfw, p_nbbw, shareid, xdata)
{
  debug0("** constructor MachSig\n");
}

MachSig::MachSig(const MachSig &m)
 : MachLin(m)
{
  debug0("** copy constructor MachSig\n");
}

MachSig::~MachSig()
{
  printf("** destructor MachSig %lx\n",(luint) this);
}

//-----------------------------------------------
// Tools
//-----------------------------------------------

void MachSig::Info(bool detailed, char *txt)
{
  if (detailed) {
    cout << "Information on sigmoidal machine" << endl;
    MachLin::Info(detailed,txt);
  }
  else {
    if (drop_out>0)
      printf("%sMachSig %d-%d, bs=%d, drop-out=%4.2f, passes=%lu/%lu", txt, idim, odim, bsize, drop_out, nb_forw, nb_backw);
    else
      printf("%sMachSig %d-%d, bs=%d, passes=%lu/%lu", txt, idim, odim, bsize, nb_forw, nb_backw);
    if (lr_coeff != 1.0) printf(", lrate-coeff=%.2f", lr_coeff);
#ifdef BLAS_CUDA
    printf(", on GPU %d", Gpu::GetCudaDevice(Gpu::GetDevice(gpu_conf)));
#endif
    tm.disp(", ");
    printf("\n");
    debug5("%s   data: %p -> %p, grad %p <- %p\n", txt, (void*)data_in, (void*)data_out, (void*)grad_in, (void*)grad_out);
  }
}

//-----------------------------------------------
// Training
//-----------------------------------------------

void MachSig::Forw(int eff_bsize, bool in_train)
{
  debug0("** MachSig Forw\n");

  tm.start();

  if (eff_bsize<=0) eff_bsize=bsize;
  MachLin::Forw(eff_bsize,in_train);

    // apply sigmoid on output
  Error("implement sigmoid\n");

  tm.stop();
}

void MachSig::Backw(const float lrate, const float wdecay, int eff_bsize)
{
  debug0("** MachSig Backw\n");
    // derivate sigmoidal activation function
    //             = grad_hidden .* ( 1 - a_hidden^2 )

  REAL *aptr = data_out;
  REAL *gptr = grad_out;

  if (eff_bsize<=0) eff_bsize=bsize;
  if (!grad_out)
    Error("MachSig::Backw(): output gradient is not set");

  tm.start();

  for (int i=0; i<odim*eff_bsize; i++) {
    REAL val = *aptr++;
    Error("implement derivative of sigmoid\n");
    *gptr=val;
  }

  tm.stop();
  MachLin::Backw(lrate, wdecay, eff_bsize);
}

