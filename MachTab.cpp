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
#include <stdlib.h>

#include "Tools.h"
#include "MachTab.h"
#include "Blas.h"

#ifdef BLAS_CUDA
# include "Gpu.cuh"
#endif

void MachTab::do_alloc()
{
  debug0("do_alloc MachTab\n");
  if (!bExternal) {
#ifdef BLAS_CUDA
    Gpu::SetConfig(gpu_conf);
    t = Gpu::Alloc(idim*odim, "memory for table look-up machine");
    debug3("    CUDA alloc table at %p, size %dx%d\n", (void*)t, idim,odim);
#else
    t = new REAL[idim*odim];
    debug3("    alloc table at %p, size %dx%d\n", (void*)t, idim,odim);
    if (!t) Error ("can't allocate memory for table look-up machine");
#endif
  }
  else {
    debug3("    reuse table at %p, size %dx%d\n", (void*)t, idim,odim);
  }
#ifdef BLAS_CUDA
    tmp_inp = new REAL[idim*bsize];
#endif
}

MachTab::MachTab(const int p_idim, const int p_odim, const int p_bsize, const ulong p_nbfw, const ulong p_nbbw, const int shareid, const bool xtable)
 : Mach(1, p_odim, p_bsize, p_nbfw, p_nbbw), Shareable(xtable, shareid), t(NULL), t_shared(NULL), t_mutex(NULL)
{
  debug1("** constructor MachTab %lx\n", (luint) this);
  if (p_idim<=0) Error("Table machine: illegal value of input dimension");
  if (p_odim<=0) Error("Table machine: illegal value of output dimension");
  idim = p_idim; // override 1 in call to Mach()

  do_alloc();

    // look-up table sharing
  t_mutex = new pthread_mutex_t;
  if (t_mutex != NULL) {
    pthread_mutex_init(t_mutex, NULL);
    int *new_t_shared = new int;
    if (new_t_shared != NULL) {
      (*new_t_shared) = 0;
      t_shared = new_t_shared;
    }
  }
    
}

MachTab::MachTab(const MachTab &m)
 : Mach(m, 1), Shareable(true, -1), t(NULL),
   t_shared(NULL), t_mutex(NULL)
{
  debug1("** copy constructor MachTab with address %lx\n", (luint) this);
  idim = m.idim; // override 1 in call to Mach()
  //bExternal = m.bExternal; //Loic: why? this should ALWAYS be true (as in initialization) 
  iShareId = m.iShareId;

  if (bExternal) {
    // set look-up table with external address
    do_alloc(); //Loic: only init tmp_inp for CUDA ??
    t = m.t;
  }
  else {
    int inc_t_shared = 0;
    if (m.t_mutex != NULL) {
      pthread_mutex_lock(m.t_mutex);
      inc_t_shared = ((m.t_shared != NULL) ? (*m.t_shared) + 1 : 0);
      if (inc_t_shared > 0) {
        (*m.t_shared) = inc_t_shared;

          // share the look-up table
        t = m.t;
        t_shared = m.t_shared;
        t_mutex = m.t_mutex;
      }
      pthread_mutex_unlock(m.t_mutex);
    }
    if (inc_t_shared <= 0)
      Error ("can't share memory for table look-up machine");
  }
}

MachTab::~MachTab()
{
  debug1("** destructor MachTab %lx\n", (luint) this);

#ifdef BLAS_CUDA
  if (tmp_inp) delete tmp_inp;
#endif

    // verify if the look-up table is shared
  if (t_mutex != NULL) {
    pthread_mutex_lock(t_mutex);
    if (t_shared != NULL) {
      if ((*t_shared) > 0) {
        debug1("*** cloned -> not freeing t %p\n", t);
        (*t_shared)--;
        pthread_mutex_unlock(t_mutex);
        return;
      }
      else {
        delete t_shared;
        t_shared = NULL;
      }
    }
  }

#ifdef BLAS_CUDA
  if (!bExternal & (t!=NULL)) cublasFree(t);
#else
  if (!bExternal & (t!=NULL)) delete [] t;
#endif
  t = NULL;

    // destroy mutex
  if (t_mutex != NULL) {
    pthread_mutex_t *old_t_mutex = t_mutex;
    t_mutex = NULL;
    pthread_mutex_unlock(old_t_mutex);
    pthread_mutex_destroy(old_t_mutex);
    delete old_t_mutex;
  }
}

void MachTab::TableConst(const REAL val)
{
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  nppsSet_32f(val,t,idim*odim);
#else
  for (int i=0; i<idim*odim; i++) t[i]=val;
#endif
}

void MachTab::TableRandom(const REAL range)
{
  REAL c=range*2.0;
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
#ifdef CURAND
  curandGenerateUniform(cuda_gen, (float*) t, idim*odim);
  Gpu::CheckError("generating random values for table look-up machine");
  nppsSubC_32f_I(0.5,t,idim*odim);
  nppsMulC_32f_I(c,t,idim*odim);
#else
  REAL * tmp = new REAL[idim * odim];
  for (int i=0; i<idim * odim; i++) tmp[i]=c*(drand48()-0.5);
  cublasSetVector(idim * odim, sizeof(REAL), tmp, 1, t, 1);
  delete [] tmp;
#endif
#else
  for (int i=0; i<idim*odim; i++) t[i]=c*(drand48()-0.5);
#endif
}

void MachTab::Info(bool detailed, char *txt)
{
  if (detailed) {
    cout << "Information on table look-up machine" << endl;
    Mach::Info(detailed,txt);
  }
  else {
    if(Mach::fileid >= file_header_version3)
	printf("%sMachTab %c%c[%d]-%d, bs=%d, passes=%lu/%lu", txt, bExternal?'s':'p', iShareId!=-1?iShareId+'0':'-', idim, odim, bsize, nb_forw, nb_backw);
    else
	printf("%sMachTab %c[%d]-%d, bs=%d, passes=%lu/%lu", txt, bExternal?'s':'p', idim, odim, bsize, nb_forw, nb_backw);
    
    if (lr_coeff != 1.0) printf(", lrate-coeff=%.2f", lr_coeff);

#ifdef BLAS_CUDA
    printf(", on GPU %d", Gpu::GetCudaDevice(Gpu::GetDevice(gpu_conf)));
#endif
    tm.disp(", ");
    printf(", LookupTable=%p", t); //DEBUG 
    printf("\n");
    debug5("%s   data: %p -> %p, grad %p <- %p\n", txt, (void*)data_in, (void*)data_out, (void*)grad_in, (void*)grad_out);
  }
}

bool MachTab::CopyParams(Mach* mach)
{
  MachTab* machtab = static_cast<MachTab*>(mach);
  if (    Mach::CopyParams(mach)
      && (machtab->bExternal == this->bExternal) ) {
#ifdef BLAS_CUDA
    Gpu::SetConfig(gpu_conf);
    Gpu::MemcpyAsync(this->t, machtab->t, idim * odim * sizeof(REAL), cudaMemcpyDeviceToDevice);
#else
    memcpy(this->t, machtab->t, idim * odim * sizeof(REAL));
#endif
    return true;
  }
  else
    return false;
}

//-----------------------------------------------
// File output
//-----------------------------------------------

void MachTab::WriteParams(ostream &of)
{
  debug0("* write params of type MachTab\n");

  Mach::WriteParams(of);
  of.write((char*) &bExternal, sizeof(int));
  of.write((char*) &iShareId, sizeof(int));
}

void MachTab::WriteData(ostream &outf) {
  int i=0, s=sizeof(REAL);
  if (bExternal) {
    debug0("* table look-up machine with external address to file\n");
    //fprintf(stderr, "* table look-up machine with external address to file\n");
    outf.write((char*) &i, sizeof(int));
    outf.write((char*) &s, sizeof(int));
  }
  else {
    debug0("* writing data of table look-up machine to file\n");
    //fprintf(stderr, "* writing data of table look-up machine to file\n");
    i=idim*odim;
    outf.write((char*) &i, sizeof(int));
    outf.write((char*) &s, sizeof(int));
#ifdef BLAS_CUDA
    REAL *local_mem=new REAL[i];
    Gpu::SetConfig(gpu_conf);
    cublasGetVector(i,CUDA_SIZE,t,1,local_mem,1);
    Gpu::CheckError("transfer of table look-up machine from GPU memory");
    outf.write((char*)local_mem,i*sizeof(REAL));
    delete [] local_mem;
#else
    outf.write((char*) t,i*sizeof(REAL));
#endif

  }
}


REAL *MachTab::WeightTable(int &idm, int &odm) {
  debug0("* dump weights under textual form from a MachTab machine\n");
  idm = idim;
  odm = odim;
	REAL *myTable = (REAL *) malloc (sizeof(REAL)*idim*odim);
#ifdef BLAS_CUDA	
  cudaMemcpy(myTable, t, idim*odim * sizeof(REAL), cudaMemcpyDeviceToHost);
#else
  memcpy(myTable, t, idim*odim * sizeof(REAL));
#endif
  return myTable;
}
	
	

//-----------------------------------------------
// File input
//-----------------------------------------------

void MachTab::ReadParams(istream &inpf, bool with_alloc)
{
  debug0("* read params of type MachTab\n");

  Mach::ReadParams(inpf, false);
  inpf.read((char*) &bExternal, sizeof(int));
  debug1(" - bExternal=%d\n", (int) bExternal);

  //This should be done for file_version 3 or greater !
  if(Mach::fileid >= file_header_version3){
    inpf.read((char*) &iShareId, sizeof(int));
    debug1(" - share-id=%d\n", (int) iShareId);
  }

  do_alloc();
}

void MachTab::ReadData(istream &inpf, size_t s, int bs)
{
  size_t se=odim*idim;
  debug1("* read data of MachTab of size %u\n", (uint)s);

  if (bExternal) {
    if (s>0) {
      ErrorN("internal error in file, table look-up machine has external address, but %u elements of data are provided\n",(uint)s);
    }
    return;	// address will be filled in by MachPar
  }
  else if (s!=se) {
    ErrorN("data block of table look-up machine has %u elements - %u were expected)",(uint) s, (uint) se);
  } 
  Mach::ReadData(inpf, 0, bs);
#ifdef BLAS_CUDA
  REAL *local_mem=new REAL[odim*idim];
  inpf.read((char*)local_mem,odim*idim*sizeof(REAL));
  debug2("CUDA: transfer %d elements for MachTab to GPU %d\n",odim*idim,Gpu::GetCudaDevice(Gpu::GetDevice(gpu_conf)));
  Gpu::SetConfig(gpu_conf);
  cublasSetVector(odim*idim,CUDA_SIZE,local_mem,1,t,1);
  Gpu::CheckError("transfer of table look-up machine to GPU memory");
  delete [] local_mem;
#else
  inpf.read((char*) t,odim*idim*sizeof(REAL));
#endif
}


//-----------------------------------------------
// Training
//-----------------------------------------------

void MachTab::Forw(int eff_bsize, bool in_train)
{
  if (!data_in)
    Error("MachTab::Forw(): input data is not set");

  debugMachInp("MachTab",data_in,1,odim,eff_bsize);

#if 0
  printf("CODES: %d%%d\n",idim,odim);
  REAL *tptr=t;
  for (int i=0; i<idim; i++) {
    printf("code %2d:", i);
    for (int o=0; o<odim; o++) printf(" %5.2f", *tptr++);
    printf("\n");
  }
#endif

  tm.start();

  if (eff_bsize<=0) eff_bsize=bsize;
  debug3("MachTab::Forw: %p -> %p, bs=%d\n",(void*)data_in,(void*)data_out,eff_bsize);

#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  Gpu::MachTabForw(eff_bsize, odim, data_in, t, data_out);
  Gpu::CheckError("MachTab::Forw - After Gpu::MachTabForw");
#else
  REAL *optr=data_out;
  for (int b=0; b<eff_bsize; b++) {
    int idx= (int) data_in[b];
    if (idx==NULL_WORD) {
        // simulate empty word: set everything to 0
      debug4("MachTab %p: b=%d, empty word    to %p, size %d\n", this, b, (void*)optr, odim);
      for (int i=0; i<odim; i++) *optr++=0.0;
    }
    else {
      debug5("MachTab %p: b=%d, memcpy idx %d to %p, size %d\n", this, b, idx, (void*)optr, odim);
      memcpy(optr,t+idx*odim,odim*sizeof(REAL));
      debug4(" partial codes: %e %e .. %e %e\n", optr[0],optr[1],optr[odim-2],optr[odim-1]);
      optr+=odim;
    }
  }
#endif

  nb_forw+=eff_bsize;
  debug0("MachTab::Forw done\n");

  tm.stop();
  debugMachOutp("MachTab",data_out,idim,odim,eff_bsize);
}


void MachTab::Backw(const float lrate, const float wdecay, int eff_bsize)
{
  debug2("MachTab::Backw: %p <- %p\n",(void*)grad_in,(void*)grad_out);
  // table[wid] = table[wid] + lrate * grad_out[wid] * data_in[wid]

  REAL lrate_bs = lr_coeff * lrate / sqrt(GetBsize());	// scale by block size !
  tm.start();

#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  if (update) {
    Gpu::MachTabBackw(lrate_bs,eff_bsize, odim, data_in, t, grad_out);
  }
    // we don't backprop to the input of a table look-up machine
  debug0("clear input grads\n");
  Gpu::MemsetAsync(grad_in, 0, eff_bsize*sizeof(REAL));
#else
  if (update) {
    REAL *gptr = grad_out;
    for (int b=0; b<eff_bsize; b++,gptr+=odim) {
      int idx= (int) data_in[b];
      if (idx==NULL_WORD) { // empty word: no weight update
        debug2("MachTab %lx: empty word at idx %d\n", (luint) this, idx);
      }
      else {
        REAL *tptr=t+idx*odim;
        debug3("b=%d idx=%d tptr=%p\n", b, idx, tptr);
        debug4("  partial grads: %e %e .. %e %e\n", gptr[0],gptr[1],gptr[odim-2],gptr[odim-1]);
        debug4("  codes        : %e %e .. %e %e\n", tptr[0],tptr[1],tptr[odim-2],tptr[odim-1]);
        AXPY(&odim,&lrate_bs,gptr,&inc1,tptr,&inc1);
        debug4("  codes updated : %e %e .. %e %e\n", tptr[0],tptr[1],tptr[odim-2],tptr[odim-1]);
      }
    }
  }

    // we don't backprop to the input of a table look-up machine
  debug0("clear input grads\n");
  for (int b=0; b<eff_bsize; b++) grad_in[b]=0.0;
#endif

  debug0("MachTab::Backw() done\n");
  tm.stop();
}

