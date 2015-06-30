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

#include "TrainerNgramClass.h"

using namespace std;

TrainerNgramClass::TrainerNgramClass(Mach* pmach, Lrate* lrate, ErrFct* perrfct,
    const char* train_fname, const char* dev_fname,
    REAL wdecay, int max_epochs, int curr_epoch)
  : TrainerNgram(pmach, lrate, perrfct, train_fname, dev_fname,
                 wdecay, max_epochs, curr_epoch, true),
    wlist(NULL), cerrfct(NULL), machclass(NULL)
{
  /*
   * In addition to the constructor of TrainerNgram, we get a wordlist from
   * Data, and set up the error function (ErrFctSoftmClassCrossEntNgram)
   * and the last layer (MachSoftmaxClass) to use an architecture derived from the
   * word classes in the word list.
   * NB: This is done in the Trainer because there is currently no way of specifying
   * that directly in the config file. We may want to add that in the future.
   */
  // get word list
  if (wlist == NULL) {
    vector<WordList> *vect_wlist = NULL;
    if (data_dev != NULL)
      vect_wlist = data_dev->GetTgtWList();
    else if (data_train != NULL)
      vect_wlist = data_train->GetTgtWList();
    if ((vect_wlist != NULL) && !vect_wlist->empty())
      wlist = &(vect_wlist->front());
  }
  if (wlist) {
    class_sizes = wlist->GetClassSizes();
  }

  // Get error function
  cerrfct = dynamic_cast<ErrFctSoftmClassCrossEntNgram*>(perrfct);
  if (cerrfct == NULL) {
    Error("In TrainerNgramClass, the error function has to be derived from ErrFctSoftmClassCrossEntNgram");
  }

  // Get output layer
  Mach* last_layer = FindFirstMatching(file_header_mtype_softmax_class, pmach);
  if (last_layer == NULL) {
    Error("In TrainerNgramClass, the last layer must contain a MachSoftmaxClass.");
  }
  machclass = dynamic_cast<MachSoftmaxClass*>(last_layer);
  if (machclass == NULL) {
    Error("Machine has mtype file_header_mtype_softmax_class, but could not be cast to MachClass*");
  }

  // Allocate buffers for classification target
  n_classes = class_sizes.size();
#ifdef BLAS_CUDA
  cudaError_t err = cudaMallocHost(&buf_class_target, n_classes*bsize*sizeof(REAL));
  if (err != cudaSuccess) {
    Error("Not able to allocate pinned host memory");
  }
  gpu_class_target = Gpu::Alloc(n_classes*bsize, "class targets in TrainerNgramClass");

  err = cudaMallocHost(&buf_class_target_info, bsize*2*sizeof(int));
  if (err != cudaSuccess) {
    Error("Not able to allocate pinned host memory");
  }
  cudaMalloc(&gpu_class_target_info, bsize*2*sizeof(int));
#else
  buf_class_target = new REAL[n_classes*bsize];
  buf_class_target_info = new int[bsize*2];
#endif

  // Make the error function aware of the MachSoftmaxClass layer, so they can interact
  cerrfct->SetUp(machclass, wlist);
}

TrainerNgramClass::TrainerNgramClass(Mach* pmach, ErrFct* perrfct, Data* data)
  : TrainerNgram(pmach, perrfct, data),
    wlist(NULL), cerrfct(NULL), machclass(NULL)
{
  vector<WordList> *vect_wlist = NULL;
  if (data_dev != NULL)
    vect_wlist = data_dev->GetTgtWList();
  if ((vect_wlist != NULL) && !vect_wlist->empty())
    wlist = &(vect_wlist->front());
  if (wlist)
    class_sizes = wlist->GetClassSizes();

  // Get error function
  cerrfct = dynamic_cast<ErrFctSoftmClassCrossEntNgram*>(perrfct);
  if (cerrfct == NULL) {
    Error("In TrainerNgramClass, the error function has to be derived from ErrFctSoftmClassCrossEntNgram");
  }

  // Get output layer
  Mach* last_layer = FindFirstMatching(file_header_mtype_softmax_class, pmach);
  if (last_layer == NULL) {
    Error("In TrainerNgramClass, the last layer must contain a MachSoftmaxClass.");
  }
  machclass = dynamic_cast<MachSoftmaxClass*>(last_layer);
  if (machclass == NULL) {
    Error("Machine has mtype file_header_mtype_softmax_class, but could not be cast to MachClass*");
  }

  // Allocate buffers for classification target
  n_classes = class_sizes.size();
#ifdef BLAS_CUDA
  cudaError_t err = cudaMallocHost(&buf_class_target, n_classes*bsize*sizeof(REAL));
  if (err != cudaSuccess) {
    Error("Not able to allocate pinned host memory");
  }
  gpu_class_target = Gpu::Alloc(n_classes*bsize, "class targets in TrainerNgramClass");

  err = cudaMallocHost(&buf_class_target_info, bsize*2*sizeof(int));
  if (err != cudaSuccess) {
    Error("Not able to allocate pinned host memory");
  }
  cudaMalloc(&gpu_class_target_info, bsize*2*sizeof(int));
#else
  buf_class_target = new REAL[n_classes*bsize];
  buf_class_target_info = new int[bsize*2];
#endif

  // Make the error function aware of the MachSoftmaxClass layer, so they can interact
  cerrfct->SetUp(machclass, wlist);
}

TrainerNgramClass::~TrainerNgramClass()
{
#ifdef BLAS_CUDA
  if (buf_class_target)
    cudaFreeHost(buf_class_target);
  if (gpu_class_target)
    cublasFree(gpu_class_target);
  if (buf_class_target_info)
    cudaFreeHost(buf_class_target_info);
  if (gpu_class_target_info)
    cudaFree(gpu_class_target_info);
#else
  if (buf_class_target)
    delete [] buf_class_target;
  if (buf_class_target_info)
    delete [] buf_class_target_info;
#endif
}

REAL TrainerNgramClass::Train()
{
  // This is, in principle, the same as in TrainerNgram::Train, with the addition
  // of additional buffers for classification.
  // We cannot simply call TrainerNgram::Train() in here, because those buffers
  // have to be filled in the inner loop. Maybe we should refactor that.

  if (!data_train) return -1;
#ifdef DEBUG
  printf("*****************\n");
  printf("TrainerNgramClass::Train():\n");
  printf(" -     data_in: %p \n", (void*) buf_input);
  printf(" -      target: %p \n", (void*) buf_target);
  printf(" -   cl_target: %p \n", (void*) buf_class_target);
  printf(" - cl_tgt_info: %p \n", (void*) buf_class_target_info);
  printf(" -    grad_out: %p \n", (void*) cerrfct->GetGrad());
  printf(" - cl_grad_out: %p \n", (void*) cerrfct->GetGradClass());
#endif

  data_train->Rewind();
  Timer ttrain;		// total training time
  ttrain.start();

  REAL log_sum=0;
  REAL class_err_sum=0;
  int i;
  nb_ex=0;

#ifdef BLAS_CUDA
  Gpu::SetConfig(mach->GetGpuConfig());
  mach->SetDataIn(gpu_input);		// we copy from buf_input to gpu_input
  cerrfct->SetTarget(gpu_target);	// we copy from buf_target to gpu_target
  cerrfct->SetTargetClassInfo(gpu_class_target, gpu_class_target_info);
#else
  mach->SetDataIn(buf_input);
  cerrfct->SetTarget(buf_target);
  cerrfct->SetTargetClassInfo(buf_class_target, buf_class_target_info);
#endif
  cerrfct->SetOutput(mach->GetDataOut());
  cerrfct->SetOutputClass(machclass->GetDataOutClass());
  mach->SetGradOut(errfct->GetGrad());
  machclass->SetGradOutClass(cerrfct->GetGradClass());

  bool data_available;
  do {
      // get a bunch of data
    int n=0;
    data_available = true;
    while (n < mach->GetBsize() && data_available) {
      data_available = data_train->Next();
      if (!data_available) break;
      for (i=0; i<iaux; i++) { // copy word indexes
        WordList::WordIndex inp = (WordList::WordIndex) data_train->input[i];
        buf_input[n*idim + i] = (REAL) wlist->MapIndex(inp, "TrainerNgramClass::Train(): input");
      }
      for (; i < idim ; i++) // copy auxiliary data
        buf_input[n * idim + i] = data_train->input[i];
      buf_target_wid[n] = wlist->MapIndex((WordList::WordIndex) data_train->target[0],
                                          "TrainerNgram::Train(): target");
      buf_target[n] = (REAL) buf_target_wid[n];

      // Compute the word class for the target
      int class_idx = wlist->GetWordInfo((int) data_train->target[0]).cl;
      buf_class_target[n] = class_idx;
      // Get offset and size info for the class
      buf_class_target_info[2*n] = wlist->MapIndex(class_idx, 0);
      buf_class_target_info[2*n+1] = class_sizes[class_idx];
      n++;
    }

    if (n>0) {
#ifdef BLAS_CUDA
      Gpu::MemcpyAsync(gpu_input, buf_input , n*idim*sizeof(REAL), cudaMemcpyHostToDevice);
      Gpu::MemcpyAsync(gpu_target, buf_target , n*1*sizeof(REAL), cudaMemcpyHostToDevice);
      Gpu::MemcpyAsync(gpu_class_target, buf_class_target, n*1*sizeof(REAL), cudaMemcpyHostToDevice);
      Gpu::MemcpyAsync(gpu_class_target_info, buf_class_target_info, n*2*sizeof(int), cudaMemcpyHostToDevice);
#endif
      mach->Forw(n,true);
      // log_sum += cerrfct->CalcGrad(n);
      REAL batch_err = cerrfct->CalcGrad(n);
      if (!isfinite(batch_err) || !isfinite(log_sum+batch_err)) {
        cerr << "Non-finite value returned by ErrFct: " << batch_err << endl
             << "nb_ex:\t" << nb_ex << endl
             << "n:\t" << n << endl;
        int odim = mach->GetOdim();
#ifdef BLAS_CUDA
        REAL* gpu_output = mach->GetDataOut();
        REAL* buf_output = new REAL[n*odim];
        cudaMemcpy(buf_output, gpu_output, n*odim*sizeof(REAL), cudaMemcpyDeviceToHost);
        REAL* gpu_class_output = machclass->GetDataOutClass();
        REAL* buf_class_output = new REAL[n*n_classes];
        cudaMemcpy(buf_class_output, gpu_class_output, n*n_classes*sizeof(REAL), cudaMemcpyDeviceToHost);
#else
        REAL* buf_output = mach->GetDataOut();
        REAL* buf_class_output = machclass->GetDataOutClass();
#endif
        for (int i=0; i<n; i++) {
          if (!isfinite(buf_output[i*odim + (int) buf_target[i]]) || !isfinite(buf_class_output[i*n_classes + (int) buf_class_target[i]])) {
            cerr << "input:\t";
            for (int j=0; j<iaux; j++)
            {
              char* w = wlist->GetWordInfoMapped((int) buf_input[i*idim + j]).word;
              if (w)
                cerr << string(w) << " ";
            }
            cerr << endl
                 << "target:\t";
            char* w=wlist->GetWordInfoMapped((int) buf_target[i]).word;
            if (w == NULL)
              cerr << "NULL";
            else
              cerr << string(w);

            cerr << endl
                 << "tgt idx:\t" << buf_target[i] << endl
                 << "tgt prb:\t" << buf_output[i*odim + (int) buf_target[i]] << endl
                 << "cl idx:\t" << buf_class_target[i] << endl
                 << "cl prb:\t" << buf_class_output[i*n_classes + (int) buf_class_target[i]] << endl;
          }
        }
#ifdef BLAS_CUDA
        delete [] buf_output;
        delete [] buf_class_output;
#endif
        Error("Non-finite cost value");
      }
      log_sum += batch_err;
      class_err_sum += cerrfct->CalcWordClassError(n);
      lrate->UpdateLrateOnForw(mach->GetNbForw());
      mach->Backw(lrate->GetLrate(), wdecay, n);
    }

    nb_ex += n;
  } while (data_available);
#ifdef BLAS_CUDA
  Gpu::StreamSynchronize();
#endif

  ttrain.stop();
  ttrain.disp(" - training time: ");
  printf("\n");
  printf(" - word class error: %.2f%%\n", class_err_sum * 100 / nb_ex);

  // Remove class target buffer from error function, so outdated information
  // is not used.
  cerrfct->SetTargetClassInfo(NULL, NULL);

  if (nb_ex>0) return exp(-log_sum / (REAL) nb_ex);  // return perplexity

  return -1;
}

REAL TrainerNgramClass::TestDev(char *fname)
{
  // This is, in principle, the same as in TrainerNgram::TestDev, with the addition
  // of additional buffers for classification.
  // We cannot simply call TrainerNgram::TestDev() in here, because those buffers
  // have to be filled in the inner loop. Maybe we should refactor that.
  if (!data_dev) return -1;

  ofstream fs;
  REAL *log_probas=NULL;
  if (fname) {
#ifdef BLAS_CUDA
    Error("Dumping probability stream into file is not yet implemented for GPU cards\n");
#else
    cout << " - dumping log probability stream to file '" << fname << "'" << endl;
    fs.open(fname,ios::out);
    CHECK_FILE(fs,fname);
    log_probas = new REAL[bsize];
#endif
  }

  int i, nb_ex_dev=0;
  REAL log_sum=0;
  REAL class_err_sum=0;
  data_dev->Rewind();

#ifdef BLAS_CUDA
  Gpu::SetConfig(mach->GetGpuConfig());
  mach->SetDataIn(gpu_input);		// we copy from buf_input to gpu_input
  cerrfct->SetTarget(gpu_target);	// we copy from buf_target to gpu_target
  cerrfct->SetTargetClassInfo(gpu_class_target, gpu_class_target_info);
#else
  mach->SetDataIn(buf_input);
  cerrfct->SetTarget(buf_target);
  cerrfct->SetTargetClassInfo(buf_class_target, buf_class_target_info);
#endif
  cerrfct->SetOutput(mach->GetDataOut());
  cerrfct->SetOutputClass(machclass->GetDataOutClass());

  bool data_available;
  do {
      // get a bunch of data
    int n=0;
    data_available = true;
    while (n < mach->GetBsize() && data_available) {
      data_available = data_dev->Next();
      if (!data_available) break;
      for (i=0; i<iaux; i++) { // copy word indexes
        buf_input[n*idim + i] = (REAL) wlist->MapIndex(
            (WordList::WordIndex) data_dev->input[i],
            "TrainerNgramClass::TestDev(): input");
      }
      for (; i < idim ; i++) // copy auxiliary data
        buf_input[n * idim + i] = data_dev->input[i];
      buf_target_wid[n] = wlist->MapIndex(
          (WordList::WordIndex) data_dev->target[0],
          "TrainerNgramClass::TestDev(): target");
      buf_target[n] = (REAL) buf_target_wid[n];

      // Compute the word class for the target
      int class_idx = wlist->GetWordInfo((int) data_dev->target[0]).cl;
      buf_class_target[n] = class_idx;
      // Get offset and size info for the class
      buf_class_target_info[2*n] = wlist->MapIndex(class_idx, 0);
      buf_class_target_info[2*n+1] = class_sizes[class_idx];
      n++;
    }

      // process the bunch
    if (n>0) {
#ifdef BLAS_CUDA
      Gpu::MemcpyAsync(gpu_input, buf_input , n*idim*sizeof(REAL), cudaMemcpyHostToDevice);
      Gpu::MemcpyAsync(gpu_target, buf_target , n*1*sizeof(REAL), cudaMemcpyHostToDevice);
      Gpu::MemcpyAsync(gpu_class_target, buf_class_target, n*1*sizeof(REAL), cudaMemcpyHostToDevice);
      Gpu::MemcpyAsync(gpu_class_target_info, buf_class_target_info, n*2*sizeof(int), cudaMemcpyHostToDevice);
#endif
      mach->Forw(n,false); 
      log_sum += errfct->CalcValue(n);
      class_err_sum += cerrfct->CalcWordClassError(n);
      if (fname) {
          // dump the log probas for all words in the current minibatch
        errfct->CalcValueBatch(n, log_probas);
        for (int ni=0; ni<n; ni++) fs << log_probas[ni] << endl;
      }
    }

    nb_ex_dev += n;
  } while (data_available);

  if (fname) fs.close();
  if (log_probas) delete [] log_probas;

  // Remove class target buffer from error function, so outdated information
  // is not used.
  cerrfct->SetTargetClassInfo(NULL, NULL);

  REAL px = (nb_ex_dev>0) ? exp(-log_sum / (REAL) nb_ex_dev) : -1;
  REAL class_err_pc = (nb_ex_dev>0) ? (class_err_sum * 100 / nb_ex_dev) : -1;
  printf(" - %d %d-gram requests, wcl_err=%.2f%%, ln_sum=%.2f, overall px=%.2f\n", nb_ex_dev, iaux+1, class_err_pc, log_sum, px);

  return px;
}
