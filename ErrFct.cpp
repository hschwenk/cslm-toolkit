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
#include "ErrFct.h"
#include "Blas.h"

ErrFct::ErrFct (Mach &mach)
 : dim(mach.GetOdim()), bsize(mach.GetBsize()), 
   output(mach.GetDataOut()), target(NULL)
{
#ifdef BLAS_CUDA
  gpu_conf = mach.GetGpuConfig();
  Gpu::SetConfig(gpu_conf);
  grad = Gpu::Alloc(dim*bsize, "gradient in Error Function");
#else
  grad = new REAL[dim*bsize];
#endif
  debug4("*** ErrFct() constructor, mach=%p, allocated %dx%d for gradient at %p\n",(void*)&mach,bsize,dim,(void*)grad);
}

ErrFct::ErrFct (const ErrFct &efct)
 : dim(efct.dim), bsize(efct.bsize), 
   output(efct.output), target(efct.target)
{
#ifdef BLAS_CUDA
  gpu_conf = efct.gpu_conf;
  Gpu::SetConfig(gpu_conf);
  grad = Gpu::Alloc(dim*bsize, "gradient in Error Function");
#else
  grad = new REAL[dim*bsize];
#endif
  debug3("*** ErrFct() copy constructor, allocated %dx%d for gradient at %p\n",bsize,dim,(void*)grad);
}

//**************************************************************************************

REAL ErrFct::CalcValue(int eff_bsize)
{ 
  Error("ErrFct::CalcValue() should be overriden\n");
  return 0.0;
}

void ErrFct::CalcValueBatch(int eff_bsize, REAL *res)
{ 
  Error("ErrFct::CalcValueBatch() should be overriden\n");
}

void ErrFct::CalcMax(int eff_bsize, REAL *res, int *idx)
{ 
  Error("ErrFct::CalcMax() should be overriden\n");
}

REAL ErrFct::CalcGrad(int eff_bsize)
{ 
  Error("ErrFct::CalcGrad() should be overriden\n");
  return 0.0;
}
