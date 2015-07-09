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
#include "MachJoin.h"

#ifdef BLAS_CUDA
#include "Gpu.cuh"
#endif

/*
 * we allocate a global input gradient but it is not used
 */
void MachJoin::do_alloc(bool alloc_data_out)
{
  debug2("do_alloc MachJoin %d x %d\n",idim,odim);
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  if (alloc_data_out) {
    if (data_out) cublasFree(data_out);
    data_out = Gpu::Alloc(odim*bsize, "output data of join machine");
    debug1("ALLOCATE output data [%d] of first machine in MachJoin\n",odim);
  }
  if (grad_in) cublasFree(grad_in);
  grad_in = Gpu::Alloc(idim*bsize, "input gradient of join machine");
  debug2(" - CUDA grad_in  alloc %lu bytes at %p\n",sizeof(REAL)*idim*bsize,(void*) grad_in);

  if (NULL == gpu_dev_data_out)
      gpu_dev_data_out = Gpu::Alloc(odim*bsize*sizeof(REAL), "MachJoin::Forw tmp for AXPY");

  // If more than 1 device is used, allocate (on the main device) a buffer
  // large enough to contain one input minibatch for any of the sub-machines,
  // before it is copied to the sub-machine's device.
  if (sub_input_tmp)
    cudaFree(sub_input_tmp);
  if (Gpu::GetDeviceCount() > 1) {
    Gpu::SetConfig(gpu_conf);
    // use the max of machine's idim, so it can be used for any of the machines
    int max_idim = 0;
    for (uint m=0; m<machs.size(); m++) {
      int m_idim = machs[m]->GetIdim();
      if (m_idim > max_idim) {
        max_idim = m_idim;
      }
    }
    Gpu::CheckError("before alloc sub_input_tmp");
    sub_input_tmp = Gpu::Alloc(max_idim*bsize, "tmp buffer for input data");
  }
  else {
    sub_input_tmp = NULL;
  }

#else
  if (alloc_data_out) {
    if (data_out) delete [] data_out;
    data_out = (odim*bsize>0) ? new REAL[odim*bsize] : NULL;
    debug1("ALLOCATE output data [%d] of first machine in MachJoin\n",odim);
    // Allocate a buffer that will contain the output gradient passed to
    // each sub-machine. This is needed because the sub-machine's call
    // to Backw() can destroy the content of their grad_out buffer,
    // so we have to pass a copy.
    grad_out_copy = (odim*bsize>0) ? new REAL[odim*bsize] : NULL;
    debug1("ALLOCATE buffer for a copy of output grad [%d] in MachJoin\n",odim);
  }
  if (grad_in) delete [] grad_in;
  grad_in = (idim*bsize>0) ? new REAL[idim*bsize] : NULL;
  debug2(" - grad_in  alloc %lu bytes at %p\n",sizeof(REAL)*idim*bsize,(void*) grad_in);
#endif
}

void MachJoin::do_delete()
{
#ifdef BLAS_CUDA
  if (grad_in) cublasFree(grad_in);

  // free local copies of grad_out
  for (vector<Mach*>::iterator it = machs.begin(); it!=machs.end(); ++it) {
    cudaFree((*it)->GetDataIn());
    (*it)->SetDataIn(NULL);
    cudaFree((*it)->GetGradOut());
    (*it)->SetGradOut(NULL);
  }

  // free local data_out buffer
  if (NULL != gpu_dev_data_out) {
    cudaFree(gpu_dev_data_out);
    gpu_dev_data_out = NULL;
  }
  data_out = NULL;
  if (sub_input_tmp) {
    cudaFree(sub_input_tmp);
    sub_input_tmp = NULL;
  }
#else
  if (grad_in) delete [] grad_in;

  // free grad_out_copy
  if (grad_out_copy)
  {
    delete [] grad_out_copy;
    grad_out_copy = NULL;
  }
#endif
  grad_in = NULL;
}


MachJoin::MachJoin()
 : MachMulti()
{
  debug0("** constructor MachJoin\n");
#ifdef BLAS_CUDA
  gpu_dev_data_out = NULL;
  sub_input_tmp = NULL;
#endif
}

MachJoin::MachJoin(const MachJoin &m)
 : MachMulti(m)
{
  debug0("** copy constructor MachJoin\n");
#ifdef BLAS_CUDA
  gpu_dev_data_out = NULL;
  sub_input_tmp = NULL;
#endif
}

MachJoin::~MachJoin()
{
  debug0("** destructor MachJoin\n");
  do_delete();
}

MachJoin *MachJoin::Clone()
{
  MachJoin *m = new MachJoin(*this);
  if (m != NULL)
    m->CloneSubmachs(*this);
  return m;
}
 
void MachJoin::MachAdd(Mach *new_mach)
{
  if (machs.empty()) {
    debug0("** add first element to join machine\n");
    machs.push_back(new_mach);
	// think about freeing memory
    idim=new_mach->GetIdim();
    odim=new_mach->GetOdim();
    bsize=new_mach->GetBsize();
    data_in=NULL; // will be set by MachJoin::SetDataIn()
    grad_in = NULL;
    grad_out = NULL;
  }
  else {
    debug0("** add new element to join machine\n");
    if (bsize!=new_mach->GetBsize())
      Error("bunch size of new join machine does not match");
    if (odim!=new_mach->GetOdim())
      Error("output dimension of new join machine does not match");
    machs.push_back(new_mach);

      // resize input gradient 
    idim += new_mach->GetIdim();
  }
  do_alloc(machs.size() == 1);
#ifdef BLAS_CUDA
  Gpu::SetConfig(new_mach->GetGpuConfig());
  // Always allocate input buffer, as data_in does not have the right layout
  new_mach->SetDataIn(Gpu::Alloc(new_mach->GetIdim()*bsize, "input data of joined submachine"));
  // Always allocate buffer for a local copy of grad_out, as it may be
  // overwritten when calling Back(). We need one such copy for each machine
  // on the GPU, so we can do all copies at the beginning of Back(), and avoid
  // forcing synchronization.
  new_mach->SetGradOut(Gpu::Alloc(odim*bsize, "copy of grad_out for a submachine of MachJoin"));
  Gpu::SetConfig(gpu_conf);
#else
  new_mach->SetDataIn(new REAL[new_mach->GetIdim()*bsize]);
  new_mach->SetGradOut(NULL); // will be set before first Backw()
#endif

  activ_forw.push_back(true);
  activ_backw.push_back(true);
}

Mach *MachJoin::MachDel()
{
  if (machs.empty()) {
    Error("impossible to delete element from join machine: is already empty");
  }

  Mach *del_mach=machs.back();
  machs.pop_back();

  if (machs.empty()) {
    idim=odim=bsize=0;
    do_delete();
    data_in = NULL;
  }
  else {
      // resize input
    idim -= del_mach->GetIdim();
  }

  activ_forw.pop_back();
  activ_backw.pop_back();

  // free local data_in buffer of submachine
  REAL* loc_data_in = del_mach->GetDataIn();
  if (NULL != loc_data_in) {
#ifdef BLAS_CUDA
    cudaFree(loc_data_in);
#else
    delete [] loc_data_in;
#endif
  }

#ifdef BLAS_CUDA
  // free local copy of grad_out
  cudaFree(del_mach->GetGradOut());
#endif

  del_mach->SetDataIn(NULL);
  del_mach->SetGradOut(NULL);
  return del_mach;
}

// set pointer of input data
void MachJoin::SetDataIn(REAL *data)
{
  // Simply set the pointer. The data will be copied in Forw().
  data_in=data;
}

// set pointer of output gradient
void MachJoin::SetGradOut(REAL *data)
{
  grad_out=data;

  // Do not make the sub-machines' grad_out point to this->grad_out,
  // as calling their Backw() method can overwrite the content of
  // their grad_out. Instead, we will use:
  // - grad_out_copy if the submachine is on CPU
  // - pre-allocated memory already in the submachine's grad_out if on GPU
#ifdef BLAS_CUDA
  // Everything is already allocated.
#else
  for (unsigned int m=0; m<machs.size(); m++)
    machs[m]->SetGradOut(grad_out_copy);
#endif
}


//-----------------------------------------------
// File output
//-----------------------------------------------

void MachJoin::ReadData(istream &inpf, size_t s, int bs)
{
  debug0("* read data of MachJoin\n");
#ifdef BLAS_CUDA
  if (s!=machs.size())
    ErrorN("data block of join machine has %zu machines (%zu were expected)", s, machs.size());

  idim=0;
  for (vector<Mach*>::iterator it = machs.begin(); it!=machs.end(); ++it) {
    Gpu::NewConfig();
    (*it) = Mach::Read(inpf, bs);
    idim += (*it)->GetIdim();
  }
#else
  MachMulti::ReadData(inpf, s, bs);

    // get dimensions
  idim=0;
  for (uint m=0; m<machs.size(); m++) idim += machs[m]->GetIdim();
#endif
  odim = machs[0]->GetOdim();
  bsize = machs[0]->GetBsize();

   // allocate memory
  do_delete();
  do_alloc(true);

  for (uint m=0; m<machs.size(); m++) {
#ifdef BLAS_CUDA
    Gpu::SetConfig(machs[m]->GetGpuConfig());
    machs[m]->SetDataIn(Gpu::Alloc(machs[m]->GetIdim()*bsize, "input data of joined submachine"));
    machs[m]->SetGradOut(Gpu::Alloc(odim*bsize, "copy of grad_out for a submachine of MachJoin"));
#else
    machs[m]->SetDataIn(new REAL[machs[m]->GetIdim()*bsize]);
    machs[m]->SetGradOut(NULL); // will be set in MachJoin::SetGradOut()
#endif
  }
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
#endif
}

//
// Tools
//

void MachJoin::Info(bool detailed, char *txt)
{
  if (detailed) {
    cout << "Information on join machine" << endl;
    MachMulti::Info(detailed);
  }
  else {
    printf("%sJoin machine %d-%d, bs=%d, passes=%lu/%lu", txt, idim, odim, bsize, nb_forw, nb_backw);
    tm.disp(", ");
    printf("\n");
    debug5("%s   data: %p -> %p, grad %p <- %p\n", txt, (void*)data_in, (void*)data_out, (void*)grad_in, (void*)grad_out);
    char ntxt[512];
    sprintf(ntxt,"%s  ", txt);
    for (unsigned int i=0; i<machs.size(); i++) machs[i]->Info(detailed, ntxt);
  }
  printf("%stotal number of parameters: %lu (%d MBytes)\n", txt, GetNbParams(), (int) (GetNbParams()*sizeof(REAL)/1048576));
}

// forward pass for all machines and average output into cumulated output
void MachJoin::Forw(int eff_bsize, bool in_train)
{
  debug4("** MachJoin::Forw: %p[%d] -> %p[%d]\n",(void*)data_in,idim,(void*)data_out,odim);
  if (machs.empty())
    Error("called Forw() for an empty join machine");

  debugMachInp("MachJoin",data_in,idim,odim,eff_bsize);

  tm.start();
  if (eff_bsize<=0) eff_bsize=bsize;
  int nb_activ=0;

  // The memory layout of data_in is NOT suited for the individual machines,
  // as they need one contiguous block of memory, without strides between
  // the rows.
  // Mem layout of "data_in":
  //    part1, part2, ..., partN,  # 1st example
  //    part1, part2, ..., partN,  # 2nd example
  //    ...,
  //    part1, part2, ..., partN  # eff_bsize-th example
  //
  // where "partI" is a vector representing the part of an example
  // that goes into machine I.
  //
  // Mem layout needed by the first sub-machine:
  //    part1, # 1st example
  //    part1, # 2nd example
  //    ...,
  //    part1  # eff_bsize-th example
  //
  // So we need to copy the data into the input memory buffer of the N
  // sub-machines, which is contiguous and already allocated (see MachAdd).
  REAL *iptr=data_in;

  debug2("MachJoin::Forw: copying input into individual machines input buffers - iptr=%p, idim=%d\n", iptr, idim);
#ifdef BLAS_CUDA
  Gpu::StreamSynchronize();
#endif
  for (unsigned int m=0; m<machs.size(); m++) {
    int m_idim = machs[m]->GetIdim();
    debug3("  machine: %d, ptr=%p, m_idim=%d\n", m, machs[m]->GetDataIn(), m_idim);
    if (activ_forw[m]) {
#ifdef BLAS_CUDA
      // Use Gpu::Memcpy2DAsync, which does strided copies in just one call
      Gpu::SetConfig(machs[m]->GetGpuConfig());
      Gpu::Memcpy2DAsync(machs[m]->GetDataIn(), m_idim*sizeof(REAL),
                   iptr, idim*sizeof(REAL),
                   m_idim*sizeof(REAL), eff_bsize,
                   cudaMemcpyDeviceToDevice);
#else
      // On CPU, calling memcpy in a loop is fast enough
      for (int i=0; i<eff_bsize; i++) {
        memcpy(machs[m]->GetDataIn() + i*m_idim,
               iptr + i*idim, m_idim*sizeof(REAL));
      }
#endif
    }
    iptr += m_idim;
  }

  REAL normf = 1.0f;
  int size = odim*eff_bsize;
  int inc1 = 1;
#ifdef BLAS_CUDA
  // Forward all machines
  for (unsigned int m=0; m<machs.size(); m++) {
    if (activ_forw[m]) {
      nb_activ++;
      machs[m]->Forw(eff_bsize,in_train);
      Gpu::CheckError("MachJoin::Forw after sub-mach->Forw()");
    }
    else {
      debug1("  MachJoin[%d]: forw deactivated\n",m);
    }
  }
  // Transfer everything to master GPU and accumulate in data_out
  // We will use gpu_dev_data_out for buffer.
  size_t cur_dev = Gpu::GetDevice(gpu_conf);
  REAL* buf_out;
  bool first_act = true;
  for (unsigned int m=0; m<machs.size(); m++) {
    if (activ_forw[m]) {
      size_t mach_conf = machs[m]->GetGpuConfig();
      Gpu::SetConfig(mach_conf);
      if (Gpu::GetDevice(mach_conf) == cur_dev)
        buf_out = machs[m]->GetDataOut();
      else {
        buf_out = gpu_dev_data_out;
        Gpu::MemcpyAsync(gpu_dev_data_out, machs[m]->GetDataOut(),
                   size*sizeof(REAL), cudaMemcpyDeviceToDevice);
        Gpu::StreamSynchronize();
        Gpu::SetConfig(gpu_conf);
      }
      if (first_act) {
        first_act = false;
        Gpu::MemcpyAsync(data_out, buf_out, size*sizeof(REAL), cudaMemcpyDeviceToDevice);
      }
      else {
        AXPY(size, normf, buf_out, inc1,
             data_out, inc1);
        Gpu::CheckError("MachJoin::Forw after accumulation AXPY");
      }
      Gpu::StreamSynchronize();
    }
  }
  Gpu::SetConfig(gpu_conf);
#else
  for (unsigned int m=1; m<machs.size(); m++) {
    if (activ_forw[m]) {
      nb_activ++;
      machs[m]->Forw(eff_bsize,in_train);
      AXPY(&size, &normf, machs[m]->GetDataOut(), &inc1, data_out, &inc1);
    }
    else {
      debug1("  MachJoin[%d]: forw deactivated\n",m);
    }
  }
#endif

    // normalize by number of active machines
    // TODO: make that an option
  if (nb_activ>0) {
    REAL normf = 1.0 / (REAL) nb_activ;
#ifdef BLAS_CUDA
    SCAL(size, normf, data_out, inc1);
#else
    SCAL(&size, &normf, data_out, &inc1);
#endif
  }

  nb_forw += eff_bsize; 
  debug0("MachJoin::Forw: done\n");

  tm.stop();
  debugMachOutp("MachJoin",data_out,idim,odim,eff_bsize);
}


  // backward pass for all machines
  // everything is already chained correctly
  // WARNING: the gradient wrt the input (grad_in) is NOT forwarded to the
  // layer below. This only works if MachJoin is the FIRST layer, directly
  // above the input.
void MachJoin::Backw(const float lrate, const float wdecay, int eff_bsize)
{
  debug4("** MachJoin::Backw: %p[%d] <- %p[%d]\n",(void*)grad_in,idim,(void*)grad_out,odim);
  if (machs.empty())
    Error("called Backw() for an empty join machine");
  if (eff_bsize<=0) eff_bsize=bsize;
 
  tm.start();

  debug4("** MachJoin::Backw:  %p[%d] <- %p[%d]\n", (void*) grad_in, idim, (void*) grad_out, odim);

#ifdef BLAS_CUDA
  // copy grad_out to each submachine's local buffer first
  Gpu::StreamSynchronize();
  for (unsigned int m=0; m<machs.size(); m++) {
    if (activ_backw[m]) {
      debug2("MachJoin::Backw: Copying grad_out to buffer of machine %d, at %p\n", m, machs[m]->GetGradOut());
      Gpu::SetConfig(machs[m]->GetGpuConfig());
      Gpu::MemcpyAsync(machs[m]->GetGradOut(), grad_out, odim*eff_bsize*sizeof(REAL),
                 cudaMemcpyDeviceToDevice);
    }
  }
#endif

  for (unsigned int m=0; m<machs.size(); m++) {
#ifndef BLAS_CUDA
    debugMachOutp("MachJoin::Backw: grad_out", grad_out, idim, odim, eff_bsize);
#endif

    if (activ_backw[m]) {
#ifndef BLAS_CUDA
      debugMachOutp("MachJoin::Backw: machs[m]->GetGradOut() before copy", machs[m]->GetGradOut(), idim, odim, eff_bsize);
      // copy the current output gradient to machs[m]->GetGradOut(),
      // so that each sub-machine can work on a brand new copy of grad_out,
      // without risking overwriting it.
      // For CPU Machines grad_out_copy will be used, and re-initialized here.
      // For GPU machines, all copies were done in advance in the loop above.
      memcpy(machs[m]->GetGradOut(), grad_out, odim*eff_bsize*sizeof(REAL));

      debugMachOutp("MachJoin::Backw: machs[m]->GetGradOut() after copy", machs[m]->GetGradOut(), idim, odim, eff_bsize);
#endif

      machs[m]->Backw(lrate,wdecay,eff_bsize);

#ifndef BLAS_CUDA
      debugMachOutp("MachJoin::Backw: machs[m]->GetGradOut() after machs[m]->Backw()", machs[m]->GetGradOut(), idim, odim, eff_bsize);
#endif
    }
    else {
      debug1("  MachJoin[%d]: backw deactivated\n",m);
    }
  }
#ifdef BLAS_CUDA
  // synchronize to all streams
  for (unsigned int m=0; m<machs.size(); m++) {
    if (activ_backw[m]) {
      Gpu::SetConfig(machs[m]->GetGpuConfig());
      Gpu::StreamSynchronize();
    }
  }
  Gpu::SetConfig(gpu_conf);	// reset to master GPU
#endif
  nb_backw += eff_bsize; 
  tm.stop();
}

