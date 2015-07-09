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
#include <stdlib.h>

#include "Tools.h"
#include "MachLin.h"
#include "Blas.h"
#ifdef CUDA
#  include "Gpu.cuh"
#endif

void MachLin::do_alloc()
{
  debug0("do_alloc MachLin\n");
  if(!bExternal){
#ifdef BLAS_CUDA
  debug3("*** CUDA do_alloc MachLin %d x %d on GPU %d\n", idim,odim,Gpu::GetCudaDevice(Gpu::GetDevice(gpu_conf)));
  b = Gpu::Alloc(odim, "bias of linear machine");
  w = Gpu::Alloc(idim*odim, "weights of linear machine");
  debug1("***  bias=%p\n",b);
  debug1("***  weights=%p\n",w);
#else
  debug2("*** constructor MachLin %d x %d\n", idim,odim);
  if (odim>0) {
    b = new REAL[odim];
    if (!b) Error ("can't allocate memory for bias of linear machine");
  }
  else b=NULL;
  debug1("***  bias=%p\n",b);
  if (idim*odim>0) {
    w = new REAL[idim*odim];
    if (!w) Error ("can't allocate memory for weights of linear machine");
  }
  else w=NULL;
  debug1("***  weights=%p\n",w);
#endif
  }
}

MachLin::MachLin(const int p_idim, const int p_odim, const int p_bsize, const ulong p_nbfw, const ulong p_nbbw, const int shareid, const bool xdata)
 : Mach(p_idim, p_odim, p_bsize, p_nbfw, p_nbbw), Shareable(xdata, shareid), bw_shared(NULL), bw_mutex(NULL)
{
#ifdef BLAS_CUDA
  debug3("*** CUDA constructor MachLin %d x %d on GPU %d\n", idim,odim,Gpu::GetCudaDevice(Gpu::GetDevice(gpu_conf)));
#endif
  do_alloc();
    // initialize clipping
  clip_w = clip_gradw = clip_gradb = 0;
 
    // biases and weights sharing
  bw_mutex = new pthread_mutex_t;
  if (bw_mutex != NULL) {
    pthread_mutex_init(bw_mutex, NULL);
    int *new_bw_shared = new int;
    if (new_bw_shared != NULL) {
      (*new_bw_shared) = 0;
      bw_shared = new_bw_shared;
    }
  }
}

MachLin::MachLin(const MachLin &m)
 : Mach(m), Shareable(true, -1), b(NULL), w(NULL), bw_shared(NULL), bw_mutex(NULL)
{
  debug0("*** copy constructor MachLin\n");
  iShareId = m.iShareId;
  int inc_bw_shared = 0;
  if (m.bw_mutex != NULL) {
    pthread_mutex_lock(m.bw_mutex);
    inc_bw_shared = ((m.bw_shared != NULL) ? (*m.bw_shared) + 1 : 0);
    if (inc_bw_shared > 0) {
      (*m.bw_shared) = inc_bw_shared;

        // share the weights and biases
      b = m.b;
      w = m.w;
      bw_shared = m.bw_shared;
      bw_mutex = m.bw_mutex;
    }
    pthread_mutex_unlock(m.bw_mutex);
  }
  if (inc_bw_shared <= 0)
    Error ("can't share memory for bias and weights of linear machine");
}

/*******************************************
 *
 ********************************************/

MachLin::~MachLin()
{
  debug1("*** destructor MachLin %lx\n", (luint) this);

#ifdef BLAS_CUDA
#else
#if 0
  printf("W:\n");
  for (int od=0;od<odim;od++) {
    for (int id=0;id<idim;id++) printf(" %9.7f",w[id*odim+od]);
    printf("\n");
  }
  printf("b: ");
  for (int od=0;od<odim;od++) printf(" %9.7f",b[od]);
  printf("\n");
#endif
#endif

    // verify if biases and weights are shared
  if (bw_mutex != NULL) {
    pthread_mutex_lock(bw_mutex);
    if (bw_shared != NULL) {
      if ((*bw_shared) > 0) {
        debug2("*** cloned -> not freeing w %p and b %p\n", w, b);
        (*bw_shared)--;
        pthread_mutex_unlock(bw_mutex);
        return;
      }
      else {
        delete bw_shared;
        bw_shared = NULL;
      }
    }
  }

#ifdef BLAS_CUDA
  if (b) cublasFree(b);
  if (w) cublasFree(w);
#else
  if (b) delete [] b;
  if (w) delete [] w;
#endif
  b = w = NULL;

    // destroy mutex
  if (bw_mutex != NULL) {
    pthread_mutex_t *old_bw_mutex = bw_mutex;
    bw_mutex = NULL;
    pthread_mutex_unlock(old_bw_mutex);
    pthread_mutex_destroy(old_bw_mutex);
    delete old_bw_mutex;
  }
}

void MachLin::BiasConst(const REAL val)
{
  debug2("MachLin::BiasRandom: %d =%f\n",odim,val);
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  nppsSet_32f(val, b, odim);
#else
  for (int i=0; i<odim; i++) b[i]=val;
#endif
}

void MachLin::BiasRandom(const REAL range)
{
  REAL c=range*2.0;
  debug3("MachLin::BiasRandom: %d r=%f -> +- %f\n",odim,range,c/2.0);
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
#ifdef CURAND
  curandGenerateUniform(cuda_gen, (float*) b, odim);		// in (0,1]
  Gpu::CheckError("generating random values for biases");
  nppsSubC_32f_I(0.5,b,odim);
  nppsMulC_32f_I(c,b,odim);
#else
  REAL * tmp = new REAL[odim];
  for (int i=0; i<odim; i++) tmp[i]=c*(drand48()-0.5);
  cublasSetVector(odim, sizeof(REAL), tmp, 1, b, 1);  
  free(tmp);
#endif
#else
  for (int i=0; i<odim; i++) b[i]=c*(drand48()-0.5);
#endif
}

void MachLin::WeightsConst(const REAL val)
{
  debug3("MachLin::WeightsConst: %dx%d =%f\n",idim,odim,val);
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  nppsSet_32f(val, w, idim*odim);
#else
  for (int i=0; i<idim*odim; i++) w[i]=val;
#endif
}

void MachLin::WeightsID(const REAL scale)
{
  debug3("MachLin::WeightsID: %dx%d =%f\n",idim,odim,scale);
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  REAL * tmp = new REAL[idim * odim];
  memset(tmp, 0.0, idim*odim*sizeof(REAL));
  if (idim>odim)
    for (int x=0; x<odim; x++) tmp[x*odim+x]=scale;
  else
    for (int x=0; x<idim; x++) tmp[x*odim+x]=scale;
  cublasSetVector(idim * odim, sizeof(REAL), tmp, 1, w, 1);
  delete [] tmp;
#else
  memset(w, 0.0, idim*odim*sizeof(REAL));
  if (idim>odim) 
    for (int x=0; x<odim; x++) w[x*odim+x]=scale;
  else
    for (int x=0; x<idim; x++) w[x*odim+x]=scale;
#endif
}

void MachLin::WeightsRandom(const REAL range)
{
  REAL c=range*2.0;
  debug4("MachLin::WeightsRandom: %dx%d r=%f -> +- %f\n",idim,odim,range,c/2.0);
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
#ifdef CURAND
  curandGenerateUniform(cuda_gen, (float*) w, idim*odim);
  Gpu::CheckError("generating random values for biases");
  nppsSubC_32f_I(0.5,w,idim*odim);
  nppsMulC_32f_I(c,w,idim*odim);
#else
  REAL * tmp = new REAL[idim * odim];
  for (int i=0; i<idim * odim; i++) tmp[i]=c*(drand48()-0.5);
  cublasSetVector(idim * odim, sizeof(REAL), tmp, 1, w, 1);
  delete [] tmp;
#endif
#else
  for (int i=0; i<idim*odim; i++) w[i]=c*(drand48()-0.5);
#endif
}

void MachLin::WeightsRandomFanI(const REAL range)
{
  REAL c=2.0*range/sqrt((REAL) idim);
  debug4("MachLin::WeightsRandomFanI: %dx%d r=%f -> +- %f\n",idim,odim,range,c/2.0);
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
#ifdef CURAND
  curandGenerateUniform(cuda_gen, (float*) w, idim*odim);
  Gpu::CheckError("generating FanI random values for biases");
  nppsSubC_32f_I(0.5,w,idim*odim);
  nppsMulC_32f_I(c,w,idim*odim);
#else
  REAL * tmp = new REAL[idim * odim];
  for (int i=0; i<idim * odim; i++) tmp[i]=c*(drand48()-0.5);
  cublasSetVector(idim * odim, sizeof(REAL), tmp, 1, w, 1);
  delete [] tmp;
#endif
#else
  printf("weight init FanI=%d, range =%5.3e\n",idim, c/2.0);
  for (int i=0; i<idim*odim; i++) w[i]=c*(drand48()-0.5);
#endif
}

void MachLin::WeightsRandomFanIO(const REAL range)
{
  REAL c=2.0*range/sqrt((REAL) (idim+odim));
  debug4("MachLin::WeightsRandomFanIO: %dx%d r=%f -> +- %f\n",idim,odim,range,c/2.0);
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
#ifdef CURAND
  curandGenerateUniform(cuda_gen, (float*) w, idim*odim);
  Gpu::CheckError("generating FanIO random values for biases");
  nppsSubC_32f_I(0.5,w,idim*odim);
  nppsMulC_32f_I(c,w,idim*odim);
#else
  REAL * tmp = new REAL[idim * odim];
  for (int i=0; i<idim * odim; i++) tmp[i]=c*(drand48()-0.5);
  cublasSetVector(idim * odim, sizeof(REAL), tmp, 1, w, 1);
  delete [] tmp;
#endif
#else
  for (int i=0; i<idim*odim; i++) w[i]=c*(drand48()-0.5);
#endif
}

void MachLin::Info(bool detailed, char *txt)
{
  if (detailed) {
    cout << "Information on linear machine" << endl;
    Mach::Info(detailed,txt);
  }
  else {
    if(Mach::fileid >= file_header_version4)
	printf("%sMachLin %c%c[%d]-%d, bs=%d, passes=%lu/%lu", txt, bExternal?'s':'p', iShareId!=-1?iShareId+'0':'-', idim, odim, bsize, nb_forw, nb_backw);
    else
	printf("%sMachLin %d-%d, bs=%d, passes=%lu/%lu", txt, idim, odim, bsize, nb_forw, nb_backw);

    if (lr_coeff != 1.0) printf(", lrate-coeff=%.2f", lr_coeff);

#ifdef BLAS_CUDA
    printf(", on GPU %d", Gpu::GetCudaDevice(Gpu::GetDevice(gpu_conf)));
#endif
    tm.disp(", ");
    printf(", weights=%p, bias=%p", w, b); //DEBUG 
    tm.newline();
#ifdef BLAS_CUDA
    debug5("***   %s   cuda data: %p -> %p, grad %p <- %p\n", txt, (void*)data_in, (void*)data_out, (void*)grad_in, (void*)grad_out);
#else
    debug5("***   %s   data: %p -> %p, grad %p <- %p\n", txt, (void*)data_in, (void*)data_out, (void*)grad_in, (void*)grad_out);
#endif
  }
}

bool MachLin::CopyParams(Mach* mach)
{
  MachLin* machlin = static_cast<MachLin*>(mach);
  if (Mach::CopyParams(mach)) {
    this->nb_params  = machlin->nb_params;
    this->clip_w     = machlin->clip_w;
    this->clip_gradw = machlin->clip_gradw;
    this->clip_gradb = machlin->clip_gradb;
#ifdef BLAS_CUDA
    Gpu::MemcpyAsync(this->b, machlin->b,        odim * sizeof(REAL), cudaMemcpyDeviceToDevice);
    Gpu::MemcpyAsync(this->w, machlin->w, idim * odim * sizeof(REAL), cudaMemcpyDeviceToDevice);
#else
    memcpy(this->b, machlin->b,        odim * sizeof(REAL));
    memcpy(this->w, machlin->w, idim * odim * sizeof(REAL));
#endif
    if(Mach::fileid >= file_header_version4) {
	this->bExternal = machlin->bExternal;
	this->iShareId = machlin->iShareId;
    }
    return true;
  }
  else
    return false;
}

//-----------------------------------------------
// File output
//-----------------------------------------------

void MachLin::WriteParams(ostream &of)
{
  debug0("* write params of type MachLin\n");
  Mach::WriteParams(of);
  if(Mach::fileid >= file_header_version4) {
    //fprintf(stderr, "MachLin::WriteParams - bExternal=%d iShareId=%d\n", (int) bExternal, iShareId);
    of.write((char*) &bExternal, sizeof(int));
    of.write((char*) &iShareId, sizeof(int));
  }
}

void MachLin::WriteData(ostream &outf) {
  int i=0, s=sizeof(REAL);
  if (bExternal) {
    debug0("* MachLin with external address to file\n");
    //fprintf(stderr, " MachLin with external address to file share-id=%d\n", iShareId);
    outf.write((char*) &i, sizeof(int));
    outf.write((char*) &s, sizeof(int));
  }
  else {
      //fprintf(stderr, " MachLin with its own data : share-id=%d, size=%d (idim=%d, odim=%d)\n", iShareId, odim*idim+odim, idim, odim);
      int s=odim*idim + odim;
      outf.write((char*) &s,sizeof(int));
      s=sizeof(REAL);
      outf.write((char*) &s,sizeof(int));

#ifdef BLAS_CUDA
      Gpu::SetConfig(gpu_conf);
      REAL *local_mem=new REAL[odim*idim];
      cublasGetVector(odim*idim,CUDA_SIZE,w,1,local_mem,1);
      Gpu::CheckError("transfer of weight matrix from GPU memory");
      outf.write((char*)local_mem,odim*idim*sizeof(REAL));
      delete [] local_mem;

      local_mem=new REAL[odim];
      cublasGetVector(odim,CUDA_SIZE,b,1,local_mem,1);
      Gpu::CheckError("transfer of bias vector from GPU memory");
      outf.write((char*)local_mem,odim*sizeof(REAL));
      delete [] local_mem;
#else
      debug0("*** writing data of linear machine to file\n");
      outf.write((char*) w,odim*idim*sizeof(REAL));
      outf.write((char*) b,odim*sizeof(REAL));
#endif
  }
}

//-----------------------------------------------
// File input
//-----------------------------------------------

void MachLin::ReadParams(istream &inpf, bool with_alloc)
{
  debug0("* read params of type MachLin\n");

  Mach::ReadParams(inpf, false);
  //This should be done for file_version 3 or greater !
  if(Mach::fileid >= file_header_version4){
    inpf.read((char*) &bExternal, sizeof(int));
    debug1(" - bExternal=%d\n", (int) bExternal);
//    fprintf(stderr, " - bExternal=%d", (int) bExternal);

    inpf.read((char*) &iShareId, sizeof(int));
    debug1(" - share-id=%d\n", (int) iShareId);
  //  fprintf(stderr, " - share-id=%d\n", (int) iShareId);
  }
  //fprintf(stderr, "\n");
  do_alloc();
}

void MachLin::ReadData(istream &inpf, size_t s, int bs)
{
  size_t se=odim*idim + odim;
  debug0("*** read data of MachLin\n");
  
  if (bExternal) {
    if (s>0) {
      ErrorN("MachLin: internal error in file, linear machine has external address, but %u elements of data are provided\n",(uint)s);
    }
    return;  // address will be filled in Mach::Read
  }
  else if (s!=se) {
    ErrorN("data block of linear machine has %zu elements (%zu were expected)", s, se);
  }
  
  Mach::ReadData(inpf, 0, bs);

    // read parameters
    // TODO: error checks
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  REAL *local_mem=new REAL[odim*idim];
  inpf.read((char*)local_mem,odim*idim*sizeof(REAL));
  for (int i=0;i<idim*odim;i++) 
    if (isnan(local_mem[i])) ErrorN("NAN in weights of layer %dx%d\n",idim,odim);
  debug1("*** CUDA: transfer %d elements for weights to GPU\n",odim*idim);
  cublasSetVector(odim*idim,CUDA_SIZE,local_mem,1,w,1);
  Gpu::CheckError("transfer of weight matrix to GPU memory");
  delete [] local_mem;

  local_mem=new REAL[odim];
  inpf.read((char*)local_mem,odim*sizeof(REAL));
  for (int i=0;i<odim;i++) 
    if (isnan(local_mem[i])) ErrorN("NAN in bias of layer %dx%d\n",idim,odim);
  debug1("*** CUDA: transfer %d elements for bias to GPU\n",odim);
  cublasSetVector(odim,CUDA_SIZE,local_mem,1,b,1);
  Gpu::CheckError("transfer of bias vector to GPU memory");
  delete [] local_mem;
#else
  inpf.read((char*) w,odim*idim*sizeof(REAL));
  inpf.read((char*) b,odim*sizeof(REAL));
    // checking for bad values
  for (int i=0;i<idim*odim;i++) 
    if (isnan(w[i])) ErrorN("NAN in weights of layer %dx%d\n",idim,odim);
  for (int i=0;i<odim;i++) 
    if (isnan(w[i])) ErrorN("NAN in bias of layer %dx%d\n",idim,odim);
#if 0
cout << "\nRead from file:" << endl;
  printf("W: %dx%d\n",odim,idim);
  for (int od=0;od<odim;od++) {
    for (int id=0;id<idim;id++) printf(" %9.7f",w[id*odim+od]);
    printf("\n");
  }
  printf("b:\n");
  for (int od=0;od<odim;od++) printf(" %9.7f",b[od]);
  printf("\n");
#endif
#endif
}


//-----------------------------------------------
// Training
//-----------------------------------------------

void MachLin::Forw(int eff_bsize, bool in_train)
{
  debug1("*** MachLin Forw %p\n", (void*)this);

  tm.start();

  if (!data_in)
    Error("MachLin::Forw(): input data is not set");
  if (eff_bsize<=0) eff_bsize=bsize;

  debugMachInp("MachLin",data_in,idim,odim,eff_bsize);

#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
    // copy bias <eff_bsize> times into result matrix 
  debug5("*** CUDA: MachLin::Forw %p[%d] -> %p[%d] on GPU %d\n",data_in,idim,data_out,odim,Gpu::GetCudaDevice(Gpu::GetDevice(gpu_conf)));
  Gpu::CopyVectorToMatrix(data_out, b, eff_bsize, odim);
  call_gemm(data_out, w, data_in, 1.0, odim, eff_bsize, idim);
#else
  for (int e=0; e<eff_bsize; e++)
    memcpy(data_out+e*odim,b,odim*sizeof(REAL));
  call_gemm(data_out, w, data_in, 1.0, odim, eff_bsize, idim);
#endif
  nb_forw += eff_bsize;

  tm.stop();
  debugMachOutp("MachLin",data_out,idim,odim,eff_bsize);
}

//-----------------------------------------------
// Helper function to apply drop-out in forward pass
// This must be called after applying the output functions, e.g. MachTanh
// (we can't do this in MachLin since the output function is usually non-linear and may
//  not preserve the zero value or the scaling)
//-----------------------------------------------

void MachLin::ForwDropout(int eff_bsize, bool in_train)
{
  debug0("*** MachLin ForwDropout");

  if (drop_out<=0) return;

  if (eff_bsize<=0) eff_bsize=bsize;
  int s=eff_bsize*odim;

#ifdef BLAS_CUDA
  if (in_train) {
        // perform drop-out during training: set randomly neurones to zero
    curandGenerateUniform(cuda_gen, (float*) drop_out_rand, s);		// in (0,1]
    Gpu::CheckError("generating random values for drop-out");
#ifdef DEBUG
    { REAL buf[s];
     cublasGetVector(s,sizeof(REAL),drop_out_rand,1,buf,1);
      printf(" rand : %e %e .. %e %e\n", buf[0],buf[1],buf[s-2],buf[s-1]);
    }
#endif
    Gpu::DropOut(s, data_out, drop_out_rand, drop_out);
  }
  else {
      // perform drop-out during testing : just scale the outputs
    if (drop_out>0 && !in_train) {
      REAL scale=1.0-drop_out;
      Gpu::CublasSscal(s,scale,data_out,1);
    }
  }

#else // of BLAS_CUDA

  if (in_train) {
        // perform drop-out during training: set randomly neurones to zero
    REAL *rptr=drop_out_rand;
    REAL *optr=data_out;
      // TODO: may be it is faster to create a mask to be multiplied with a element-wise product
    for (int i=0; i<s; i++, optr++) {
      *rptr=drand48();  // memorize random values for backw pass
      if (*rptr++<drop_out) *optr = 0.0;
    }
  }
  else {
        // perform drop-out during testing: just scale the outputs
    if (drop_out>0 && !in_train) {
      REAL scale=1.0-drop_out;
      SCAL(&s,&scale,data_out,&inc1);
    }
  }
#endif
}


//-----------------------------------------------
// Backprop
//-----------------------------------------------

void MachLin::Backw(const float lrate, const float wdecay, int eff_bsize)
{
  debug2("*** MachLin Backw %p <- %p\n",(void*)grad_in,(void*)grad_out);
  static REAL real1=1.0, real0=0.0;
  static char transN='N', transT='T';
  REAL lrate_bs = lr_coeff * lrate / sqrt(GetBsize());	// scale by block size !
  REAL epsilon = 1.0 + lrate_bs * wdecay;

  if (eff_bsize<=0) eff_bsize=bsize;
  if (!grad_out)
    Error("MachLin::Backw(): output gradient is not set");

  debugMachOutp("MachLin Grad",grad_out,idim,odim,eff_bsize);
  tm.start();

#if defined(BLAS_ATLAS) || defined(BLAS_INTEL_MKL)
    // perform drop-out, set selected output gradients to zero
  if (drop_out>0.0) {
    REAL *rptr=drop_out_rand;
    REAL *gptr=grad_out;
    for (int i=0; i<eff_bsize*odim; i++, gptr++) {
      if (*rptr++<drop_out) *gptr = 0.0;
    }
  }

  if (update) {
      // update bias vector:   b = b + lrate * grad_out
      // NO weight decay
    REAL *gptr = grad_out;
    for (int e=0; e<eff_bsize; e++, gptr+=odim) {
      AXPY(&odim,&lrate_bs,gptr,&inc1,b,&inc1);
    }
  }

    // backprop gradient:   grad_in   =        w'        *   grad_out
    //                    idim x bsize = (odim x idim)'  *  odim x bsize
  GEMM (&transT, &transN, &idim, &eff_bsize, &odim,
        &real1, w, &odim, grad_out, &odim,
        &real0, grad_in, &idim);

  if (update) {
      // update weights including weight decay
      // w = lrate  *grad_out * data_in^T + epsilon * w
      // gemm (transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc )
      //                                      Go      Din            W
      //        C = alpha*A * B + beta * b
      //

    GEMM (&transN, &transT, &odim, &idim, &eff_bsize,
          &lrate_bs, grad_out, &odim, data_in, &idim,
          &epsilon, w, &odim);
  }
#else
# ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);

    // perform drop-out
  if (drop_out>0) {
    Gpu::DropOut(odim*eff_bsize, grad_out, drop_out_rand, drop_out);
  }

  if (update) {
    Gpu::BatchedAXPY(odim,lrate_bs,grad_out,1,b,1,eff_bsize);
  }
    // backprop gradient:   grad_in   =        w'        *   grad_out
    //                    idim x bsize = (odim x idim)'  *  odim x bsize
  GEMM (transT, transN, idim, eff_bsize, odim,
        real1, w, odim, grad_out, odim,
        real0, grad_in, idim);
  if (update) {
      // update weights including weight decay
      // w = lrate  *grad_out * data_in^T + epsilon * w
    GEMM (transN, transT, odim, idim, eff_bsize,
          lrate_bs, grad_out, odim, data_in, idim,
          epsilon, w, odim);
  }
# else
  Error("you must compile with BLAS_ATLAS, BLAS_INTEL_MKL or BLAS_CUDA");
# endif
#endif
  nb_backw += eff_bsize;

  tm.stop();
  debugMachInp("MachLin Grad",grad_in,idim,odim,eff_bsize);
}

void MachLin::Debug()
{
#ifdef BLAS_CUDA
  Error("MachLin::Debug(): not implemented for CUDA\n");
#else
  for (int o=0; o<odim; o++) {
    for (int i=0; i<idim; i++) {
      w[i*odim+o] = i + 1000*o;
    }
    b[o] = -o;
  }
#endif
}
