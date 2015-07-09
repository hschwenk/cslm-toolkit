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
 */

using namespace std;
#include <iostream>
#include <unistd.h>
#include <time.h>
#include <float.h>

#include "Tools.h"
#include "Mach.h"
#include "ErrFctMCE.h"
#include "Trainer.h"

Trainer::Trainer (Mach *pmach, Lrate *lrate, ErrFct *perrfct,
		  const char *train_fname, const char *dev_fname,
		  REAL p_wd, int p_maxep, int p_ep, bool alloc_target)
 : mach(pmach), lrate(lrate), errfct(perrfct),
   wdecay(p_wd), nb_ex(0), nb_epoch(p_ep),
   max_epoch(p_maxep), auxdim(0), err_train(0), err_dev(0)
{
  debug0("*** Constructor Trainer ***\n");

  idim=mach->GetIdim(); odim=mach->GetOdim(); bsize=mach->GetBsize();
  if (train_fname) {
    data_train = new Data(train_fname);

    if (idim != data_train->GetIdim())
      ErrorN("Trainer: input dimension of the training data (%d) does not match the one of the machine (%d)\n", data_train->GetIdim(), idim);

    if (data_train->GetOdim()==0) { // no targets: suppose target = input
      if (idim!=odim)
        ErrorN("Trainer: training data has no targets, dimensions must match for autoassociative training (network input=%d, output=%d)", idim, odim);
    }
    else {
      if (odim != data_train->GetOdim())
        ErrorN("Trainer: ouput dimension of the training data (%d) does not match the one of the machine (%d)\n", data_train->GetOdim(), odim);
    }
    auxdim = data_train->GetAuxdim();
  }
  else 
    data_train=NULL;

  if (dev_fname) {
    data_dev = new Data(dev_fname, data_train);
    data_dev_alloc=true;
    if (idim != data_dev->GetIdim())
      ErrorN("Trainer: input dimension of the validation data (%d) does not match the one of the machine (%d)\n", data_dev->GetIdim(), idim);

    if (odim != data_dev->GetOdim())
      Error("Trainer: output dimension of the validation data does not match the one of the machine\n");
    if (data_dev->GetOdim()==0) { // no targets: suppose target = input
      if (idim!=odim)
        ErrorN("Trainer: validation data has no targets, dimensions must match for autoassociative training (network input=%d, output=%d)", idim, odim);
    }
    else {
      if (odim != data_dev->GetOdim())
        ErrorN("Trainer: ouput dimension of the validation data (%d) does not match the one of the machine (%d)\n", data_dev->GetOdim(), odim);
    }
    int auxdim_dev = data_dev->GetAuxdim();
    if (0 >= auxdim)
      auxdim = auxdim_dev;
    else if (auxdim != auxdim_dev)
      ErrorN("Trainer: auxiliary data dimension of the validation data should be %d, found %d", auxdim, auxdim_dev);
  }
  else {
    data_dev=NULL;
    data_dev_alloc=false;
  }
  iaux = (idim - auxdim);

#ifdef BLAS_CUDA
  gpu_input = Gpu::Alloc(idim*bsize, "inputs in Trainer");
  if (alloc_target)
    gpu_target = Gpu::Alloc(odim*bsize, "targets in Trainer");
  else
    gpu_target = NULL;
  host_output = new REAL[odim*bsize];
  cudaError_t err = cudaMallocHost(&buf_input, idim*bsize*sizeof(REAL));
  if(err != cudaSuccess){
    Error("Not able to allocate pinned host memory");
  }
  if (alloc_target) {
    err = cudaMallocHost(&buf_target, odim*bsize*sizeof(REAL));
    if(err != cudaSuccess){
      Error("Not able to allocate pinned host memory");
    }
  }
  else
    buf_target = NULL;
#else
  buf_input = new REAL[idim*bsize];
  if (alloc_target)
    buf_target = new REAL[odim*bsize];
  else
    buf_target = NULL;
  // memory for the output gradient is allocated by the error function
#endif
}

//**************************************************************************************

Trainer::~Trainer()
{
  debug0("*** Destructor Trainer ***\n");
  if (data_train) delete data_train;
  if (data_dev && data_dev_alloc) {
    debug0("freeing data_dev\n");
    delete data_dev;
  }
#ifdef BLAS_CUDA
  delete [] host_output;
  if (gpu_input) cublasFree(gpu_input);
  if (gpu_target) cublasFree(gpu_target);
  if (buf_input) cudaFreeHost(buf_input);
  if (buf_target) cudaFreeHost(buf_target);

#else
  delete [] buf_input;
  delete [] buf_target;
#endif
}

//**************************************************************************************

REAL Trainer::Train()
{
#ifdef DEBUG
  printf("*****************\n");
  printf("Trainer::Train():\n");
  printf(" -  data_in: %p \n", (void*) buf_input);
  printf(" -   target: %p \n", (void*) buf_target);
  printf(" - grad_out: %p \n", (void*) errfct->GetGrad());
#endif
  data_train->Rewind();
  Timer ttrain;		// total training time
  ttrain.start();

  REAL err=0;
  nb_ex=0;

#ifdef BLAS_CUDA
  Gpu::SetConfig(mach->GetGpuConfig());
  mach->SetDataIn(gpu_input);		// we copy from buf_input to gpu_input
  errfct->SetTarget(gpu_target);	// we copy from buf_target to gpu_target
  debug1(" - gpu_input %p\n", gpu_input);
  debug1(" - gpu_target %p\n", gpu_target);
#else
  mach->SetDataIn(buf_input);
  errfct->SetTarget(buf_target);
#endif
  errfct->SetOutput(mach->GetDataOut());
  mach->SetGradOut(errfct->GetGrad());
  debug1(" - grad %p\n", errfct->GetGrad());
  debug1(" - output %p\n", mach->GetDataOut());

    // TODO: we could copy all the examples on the GPU and then split into bunches locally
  bool data_available;
  do {
      // get a bunch of data
    int n=0;
    data_available = true;
    while (n < mach->GetBsize() && data_available) {
      data_available = data_train->Next();
      if (!data_available) break;
      memcpy(buf_input  + n*idim, data_train->input, idim*sizeof(REAL));
      memcpy(buf_target + n*odim, data_train->target, odim*sizeof(REAL));
      n++;
    }

    if (n>0) {
#ifdef BLAS_CUDA
      Gpu::MemcpyAsync(gpu_input, buf_input , n*idim*sizeof(REAL), cudaMemcpyHostToDevice);
      Gpu::MemcpyAsync(gpu_target, buf_target , n*1*sizeof(REAL), cudaMemcpyHostToDevice);
#endif
      mach->Forw(n,true); 
      err += errfct->CalcGrad(n);
      lrate->UpdateLrateOnForw(mach->GetNbForw());
      mach->Backw(lrate->GetLrate(), wdecay, n);
#ifdef BLAS_CUDA
      Gpu::StreamSynchronize();
#endif
    }

    nb_ex += n;
  } while (data_available);

  ttrain.stop();
  ttrain.disp(" - training time: ");
  printf("\n");

  err /= nb_ex;
  return err;
}

//**************************************************************************************
// returns the ratio of wrong classifications
// 

REAL Trainer::TestDev(char *fname)
{
  if (!data_dev) return -1;

  ofstream fs;
  if (fname) {
#ifdef BLAS_CUDA
    Error("Dumping classification errors into file is not yet implemented for GPU cards\n");
#else
    cout << " - dumping classification errors to file '" << fname << "'" << endl;
    fs.open(fname,ios::out);
    CHECK_FILE(fs,fname);
#endif
  }

  int nb_ex_dev=0;
  REAL err=0;
 
    // always use classification error
  ErrFctMCE dev_errfct(*mach);

#ifdef BLAS_CUDA
  Gpu::SetConfig(mach->GetGpuConfig());
  mach->SetDataIn(gpu_input);		// we copy from buf_input to gpu_input
  errfct->SetTarget(gpu_target);	// we copy from buf_target to gpu_target
#else
  mach->SetDataIn(buf_input);
  errfct->SetTarget(buf_target);
#endif
  errfct->SetOutput(mach->GetDataOut());

      // TODO: we could copy all the examples on the GPU and then split into bunches locally
  bool data_available;
  data_dev->Rewind();
  do {
      // get a bunch of data
    int n=0;
    data_available = true;
    while (n < mach->GetBsize() && data_available) {
      data_available = data_dev->Next();
      if (!data_available) break;
      memcpy(buf_input  + n*idim, data_dev->input, idim*sizeof(REAL));
      memcpy(buf_target + n*odim, data_dev->target, odim*sizeof(REAL));
      n++;
    }

      // process the bunch
    if (n>0) {
#ifdef BLAS_CUDA
      Gpu::MemcpyAsync(gpu_input, buf_input , n*idim*sizeof(REAL), cudaMemcpyHostToDevice);
      Gpu::MemcpyAsync(gpu_target, buf_target , n*1*sizeof(REAL), cudaMemcpyHostToDevice);
#endif
      mach->Forw(n,false); 
      err += dev_errfct.CalcValue(n);  // always calculate classification error
    }

    if (fname) {
      for (int ni=0; ni<n; ni++)
        fs << dev_errfct.CalcValue(ni) << endl;
    }

    nb_ex_dev += n;
  } while (data_available);

  if (fname) fs.close();

  if (nb_ex_dev>0) return err/nb_ex_dev;
  return -1;
}

//**********************************************************************************
// print some information when training starts

void Trainer::StartMessage()
{
  const int hlen=256;
  char hostname[hlen];
  gethostname(hostname, hlen); hostname[hlen-1]=0;
  cout << "Starting training on host " << hostname << " pid " << getpid() << endl;
  cout << " - training on " << data_train->GetFname() << endl;
  if (data_dev) 
    cout << " - validation on " << data_dev->GetFname() << endl;
  cout << " - stopping training at " << max_epoch << " epochs" << endl;

  lrate->Info();
  cout << " - scaling learning rate by sqrt of batch size" << endl;
}

//**********************************************************************************
// simple training routine 

void Trainer::TrainAndTest (const char *fname)
{
  if (!data_train) {
    cout << "No training data specified, training impossible" << endl;
    return;
  }

#define FNAME_MACH ".mach"
#define FNAME_BEST ".best.mach"
  char fname_best[strlen(fname) + strlen(FNAME_BEST) +1];
  
  StartMessage();
  cout << " - saving machine into file: " << fname << endl;
  if (data_dev) {
    strcpy(fname_best,fname);
    char *p=strstr(fname_best, FNAME_MACH);
    if (p) *p=0;
    strcat(fname_best, FNAME_BEST);
    cout << " - saving best machine on validation data into file: " << fname_best << endl;
  }
  //mach->Info();

  REAL best_err_dev=FLT_MAX; // some huge value
  int best_epoch=0;
  while (!Converged()) {
    InfoPre();
    err_train = Train();
    InfoPost();
    if (isnan(err_train) || isnan(-err_train) ) {
      cout << "   ERROR: detected numerical errors during training" << endl;
      break;
    }
    else {
      if (! data_dev) {
        cout << " - saving current machine into file '" << fname << "'" << endl;
        ofstream fs;
        fs.open(fname,ios::binary);
        CHECK_FILE(fs,fname);
        mach->Write(fs);
        fs.close();
      }
    }

    if (data_dev) {
	//TODO: Loic: we should rewind data here so it is not required to be done on overriden TestDev()
      cout << " - starting validation ..." << endl; cout.flush();
      err_dev = TestDev();

      if (isnan(err_dev) || isnan(-err_dev) ) {
        cout << "   ERROR: detected numerical errors during validation" << endl;
        cout << " - NOT saving current machine into file '" << fname_best << "'" << endl;
        break;
      }

      if (lrate->UpdateLrateOnDev(err_dev, best_err_dev, fname_best, mach)) {
        cout << " - saving current best machine into file '" << fname_best << "'" << endl;
        ofstream fs;
        fs.open(fname_best,ios::binary);
        CHECK_FILE(fs,fname_best);
        mach->Write(fs);
        fs.close();
        best_err_dev = err_dev;
        best_epoch=nb_epoch;
      }
    }

  } // while !Converged()

  if (data_dev) {
    cout << "Training stopped, lowest error was " << best_err_dev << " at epoch " << best_epoch << endl;
  
      // save machine at the end when validation was done
    cout << " - saving final machine into file '" << fname << "'" << endl;
    ofstream fs;
    fs.open(fname,ios::binary);
    CHECK_FILE(fs,fname);
    mach->Write(fs);
    fs.close();
  }
  else {
    cout << "Training stopped" << endl;
  }
  //mach->Info();
}

//**************************************************************************************

bool Trainer::Converged ()
{
  return ((nb_epoch >= max_epoch) || lrate->StopReached());
}

//**************************************************************************************
// information before starting an epoch

void Trainer::InfoPre ()
{
  time_t now;
  time(&now); // TODO: ctime is not rentrant ! use ctime_r() instead if needed
  cout << "Starting epoch " << ++nb_epoch << " at " << ctime(&now);

  lrate->UpdateLrateOnForw(mach->GetNbForw());
  printf(" - initial unscaled lrate=%6.4e, wdecay=%6.4e\n", lrate->GetLrate(), wdecay);
}

//**************************************************************************************
// information after finishing an epoch

void Trainer::InfoPost ()
{
  cout << " - epoch finished, " << nb_ex << " examples seen, average error: " << err_train << endl;
}
