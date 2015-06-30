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

#include "Tools.h"
#include "MachSplit1.h"

#ifdef BLAS_CUDA
#include "Gpu.cuh"
#endif

MachSplit1::MachSplit1()
 : MachMulti(), grad_out_split(NULL)
{
}

MachSplit1::MachSplit1(const MachSplit1 &m)
 : MachMulti(m), grad_out_split(NULL)
{
}

MachSplit1::~MachSplit1()
{
  // data_out and grad_in will be freed by Mach::~Mach()
#ifdef BLAS_CUDA
  Error("Check setting CUDA device");
  cublasFree(grad_out_split);
#else
  delete [] grad_out_split;
#endif
}

MachSplit1 *MachSplit1::Clone()
{
  MachSplit1 *m = new MachSplit1(*this);
  if (m != NULL)
    m->CloneSubmachs(*this);
  return m;
}
 
void MachSplit1::MachAdd(Mach *new_mach)
{
  if (machs.empty()) {
    machs.push_back(new_mach);
	// think about freeing memory
    idim=new_mach->GetIdim();
    odim=new_mach->GetOdim();
    bsize=new_mach->GetBsize();
#ifdef BLAS_CUDA
    Gpu::SetConfig(gpu_conf);
    data_in=NULL; // will be set by MachSplit1::SetDataIn()
    data_out = Gpu::Alloc(odim*bsize, "first output data in split machine");
    grad_in = Gpu::Alloc(idim*bsize, "input gradient in split machine");
    grad_out_split = Gpu::Alloc(odim*bsize, "first internal output gradient in split machine");
    grad_out = NULL;
#else
    data_in=NULL; // will be set by MachSplit1::SetDataIn()
    data_out = (odim*bsize>0) ? new REAL[odim*bsize] : NULL;
    grad_in = (idim*bsize>0) ? new REAL[idim*bsize] : NULL;
    grad_out_split = (odim*bsize>0) ? new REAL[odim*bsize] : NULL;
    grad_out = NULL;
#endif
    new_mach->SetDataIn(data_in);
    new_mach->SetGradOut(NULL); // will be done in Backw()
  }
  else {
    if (bsize!=new_mach->GetBsize())
      Error("bunch size of new split machine does not match");
    if (idim!=new_mach->GetIdim())
      Error("input dimension of new split machine does not match");
    machs.push_back(new_mach);
 
      // resize output (idim does not change !)
    odim += new_mach->GetOdim();
#ifdef BLAS_CUDA
    Gpu::SetConfig(gpu_conf);
    if (data_out) cublasFree(data_out);
    data_out = Gpu::Alloc(odim*bsize, "resized output data in split machine");
    if (grad_out_split) cublasFree(grad_out_split);
    grad_out_split = Gpu::Alloc(odim*bsize, "resized internal output gradient in split machine");
#else
    if (data_out) delete [] data_out;
    data_out = (odim*bsize>0) ? new REAL[odim*bsize] : NULL;
    if (grad_out_split) delete [] grad_out_split;
    grad_out_split = (odim*bsize>0) ? new REAL[odim*bsize] : NULL;
#endif
    new_mach->SetDataIn(data_in);
    new_mach->SetGradOut(NULL); // will be done in Backw()
  }

  activ_forw.push_back(true);
  activ_backw.push_back(true);
}

Mach *MachSplit1::MachDel()
{
  if (machs.empty()) {
    Error("impossible to delete element from split machine: is already empty");
  }
  
  Mach *del_mach=machs.back();
  machs.pop_back();

  if (machs.empty()) {
    idim=odim=bsize=0;
#ifdef BLAS_CUDA
    if (data_out) cublasFree(data_out);
    if (grad_in) cublasFree(grad_in);
#else
    if (data_out) delete [] data_out;
    if (grad_in) delete [] grad_in;
#endif
    data_in=data_out=grad_in=grad_out=NULL;
  }
  else {
      // resize output
    odim -= del_mach->GetOdim();
#ifdef BLAS_CUDA
    Gpu::SetConfig(gpu_conf);
    if (data_out) cublasFree(data_out);
    data_out = Gpu::Alloc(odim*bsize, "resized output data in split machine");
    if (grad_out_split) cublasFree(grad_out_split);
    grad_out_split = Gpu::Alloc(odim*bsize, "resized internal output gradient in split machine");
#else
    if (data_out) delete [] data_out;
    data_out = (odim*bsize>0) ? new REAL[odim*bsize] : NULL;
    if (grad_out_split) delete [] grad_out_split;
    grad_out_split = (odim*bsize>0) ? new REAL[odim*bsize] : NULL;
#endif
  }

  activ_forw.pop_back();
  activ_backw.pop_back();

  return del_mach;
}

// set pointer of input data
void MachSplit1::SetDataIn(REAL *data)
{
  data_in=data;
    // all machines point on the same input
  for (unsigned int m=0; m<machs.size(); m++) machs[m]->SetDataIn(data_in);
}


//-----------------------------------------------
// File input
//-----------------------------------------------


void MachSplit1::ReadData(istream &inpf, size_t s, int bs)
{
  MachMulti::ReadData(inpf, s, bs);

     // get dimensions
  odim=0;
  for (uint m=0; m<machs.size(); m++) odim += machs[m]->GetOdim();
  idim = machs[0]->GetIdim();
  bsize = machs[0]->GetBsize();
  
    // allocate memory
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  if (data_out) cublasFree(data_out);
  data_out = (odim*bsize>0) ? Gpu::Alloc(odim*bsize, "ReadData output data in split machine") : NULL;
  if (grad_out_split) cublasFree(grad_out_split);
  grad_out_split = (odim*bsize>0) ? Gpu::Alloc(odim*bsize, "ReadData gradient output in split machine") : NULL;
  if (grad_in) cublasFree(grad_in);
  grad_in = (idim*bsize>0) ? Gpu::Alloc(idim*bsize, "ReadData gradient input in split machine") : NULL;

#else
  if (data_out) delete [] data_out;
  data_out = (odim*bsize>0) ? new REAL[odim*bsize] : NULL;
  if (grad_out_split) delete [] grad_out_split;
  grad_out_split = (odim*bsize>0) ? new REAL[odim*bsize] : NULL;
  if (grad_in) delete [] grad_in;
  grad_in = (idim*bsize>0) ? new REAL[idim*bsize] : NULL;
#endif

  for (uint m=0; m<machs.size(); m++) {
    machs[m]->SetDataIn(NULL);	// will be set before first Forw()
    machs[m]->SetGradOut(NULL);	// will be done each time in Backw() using  grad_out_split
  }
}

//
// Tools
//

void MachSplit1::Info(bool detailed, char *txt)
{
  if (detailed) {
    cout << "Information on split1 machine" << endl;
    MachMulti::Info(detailed);
  }
  else {
    printf("%sSplit1 machine %d-%d, bs=%d, passes=%lu/%lu", txt, idim, odim, bsize, nb_forw, nb_backw);
    tm.disp(", ");
    tbackw.disp(" + backw: ");
    tbackw.newline();
    char ntxt[512];
    sprintf(ntxt,"%s  ", txt);
    for (unsigned int i=0; i<machs.size(); i++) machs[i]->Info(detailed, ntxt);
  }
  printf("%stotal number of parameters: %lu (%d MBytes)\n", txt, GetNbParams(), (int) (GetNbParams()*sizeof(REAL)/1048576));
}


// forward pass for all machines and copy output into cumulated output
void MachSplit1::Forw(int eff_bsize, bool in_train)
{
  if (machs.empty())
    Error("called Forw() for an empty split machine");

  debugMachInp("MachSplit1",data_in,idim,odim,eff_bsize);
  tm.start();

  if (eff_bsize<=0) eff_bsize=bsize;

  REAL *optr=data_out;
  for (unsigned int m=0; m<machs.size(); m++) {
    if (activ_forw[m]) {
      machs[m]->Forw(eff_bsize,in_train);
        // copy output to arrange each into continuous blocks
      int codim=machs[m]->GetOdim();
      REAL * dptr = machs[m]->GetDataOut();
#ifdef BLAS_CUDA
      Gpu::CopyMatrixToMatrixStrided(optr, dptr, eff_bsize, codim, odim);
#else
      REAL *optr2=optr;
      for (int b=0; b<eff_bsize; b++) {
         memcpy(optr2, dptr+b*codim, codim*sizeof(REAL));
         optr2 += odim;
      }
#endif
      optr += codim;
    }
    else {
      Error("  MachSplit1: forw deactivated\n");
    }
  }
  nb_forw += eff_bsize;

  tm.stop();
  debugMachOutp("MachSplit1",data_out,idim,odim,eff_bsize);
}

// backward pass for all machines and cumulate gradient at input
void MachSplit1::Backw(const float lrate, const float wdecay, int eff_bsize)
{
  if (machs.empty())
    Error("called Backw() for an empty split machine");

  if (eff_bsize<=0) eff_bsize=bsize;

  debugMachOutp("MachSplit1 Grad",grad_out,idim,odim,eff_bsize);
  tbackw.start();

     // copy output gradient into temp buffer by bsize pieces for each machine

  REAL *goptr=grad_out;		// this points to a buffer in the error functions
  REAL *gsptr=grad_out_split;	// this is our internal buffer for the individual machines
  for (unsigned int m=0; m<machs.size(); m++) {
    REAL *goptr2=goptr;
    machs[m]->SetGradOut(gsptr);
    int codim=machs[m]->GetOdim();
#ifdef BLAS_CUDA
    //I don't enable the faster code here as it slow down the Forws pass!
    //I don't understand why this can happen, everything was synchronized.
    Gpu::SetConfig(machs[m]->GetGpuConfig());
    //Gpu::CopyMatrixStridedToMatrix(gsptr, goptr2, eff_bsize, codim, odim);
    //gsptr += codim * eff_bsize;
    for (int b=0; b<eff_bsize; b++) {
      Gpu::MemcpyAsync(gsptr, goptr2, codim*sizeof(REAL), cudaMemcpyDeviceToDevice);
      goptr2 += odim;
      gsptr += codim;
    }
    //cudaDeviceSynchronize();

#else
    for (int b=0; b<eff_bsize; b++) {
      memcpy(gsptr, goptr2, codim*sizeof(REAL));
      goptr2 += odim;
      gsptr += codim;
    }
#endif
    goptr += codim;
  }

  debugMachOutp("MachSplit1 Grad internal",grad_out_split,idim,odim,eff_bsize);

   // backward 1st machine
  if (activ_backw[0]) {
    machs[0]->Backw(lrate,wdecay,eff_bsize);
#ifdef BLAS_CUDA
    Gpu::MemcpyAsync(grad_in, machs[0]->GetGradIn(), idim*eff_bsize*sizeof(REAL), cudaMemcpyDeviceToDevice);
#else
    memcpy(grad_in, machs[0]->GetGradIn(), idim*eff_bsize*sizeof(REAL));
#endif
  }
  else {
      // clear the gradient so we can cumulate the following ones
#ifdef BLAS_CUDA
    Gpu::SetConfig(gpu_conf);
    Gpu::MemsetAsync(grad_in, 0.0, idim*eff_bsize*sizeof(REAL));
#else
    memset(grad_in, 0.0, idim*eff_bsize*sizeof(REAL));
#endif
  }

   // backward following machines, add gradient on existing ones
  for (unsigned int m=1; m<machs.size(); m++) {
    if (activ_backw[m]) {
      machs[m]->Backw(lrate,wdecay,eff_bsize);
      REAL * grad_ptr = machs[m]->GetGradIn();
      int size = idim*eff_bsize;
      REAL onef = 1.f;
      int one = 1;
#ifdef BLAS_CUDA
      AXPY(size, onef, grad_ptr, one, grad_in, one);
#else
      AXPY(&size, &onef, grad_ptr, &one, grad_in, &one);
#endif
    }
    else {
    }
  }

  nb_backw += eff_bsize; 

  tbackw.stop();
  debugMachInp("MachSplit1 Grad",grad_in,idim,odim,eff_bsize);
}

