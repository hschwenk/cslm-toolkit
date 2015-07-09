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
#include <algorithm>
#include <unistd.h>
#include <time.h>

#include "Tools.h"
#include "Mach.h"
#include "MachTab.h"
#include "MachPar.h"
#include "MachSeq.h"
#include "MachSplit.h"
#include "TrainerPhraseSlist.h"
#include "ErrFctSoftmCrossEntNgram.h"

#include "NBest.h" 
#include "sort.cpp" 

// activate mapping of input
// not really necessary, may only speed up calculations due to cache locality
// if you activvate this option, you must do so for all your networks
#undef TRAINER_PHASE_SLIST_MAP_INPUT

void TrainerPhraseSlist::DoConstructorWork()
{
  idim=mach->GetIdim(); odim=mach->GetOdim(); bsize=mach->GetBsize();

#ifdef BLAS_CUDA
  Gpu::SetConfig(mach->GetGpuConfig());
  gpu_input = Gpu::Alloc(idim*bsize, "inputs in Trainer");
  host_output = new REAL[odim*bsize];
#endif
  buf_target_wid = new WordID[odim*bsize];	// TODO: those are actually too big, we need tg_nbphr*bsize ??
  buf_target_ext = new WordID[odim*bsize];
  buf_target_in_blocks = new REAL[odim*bsize];

    // set up vector to outputs of the target phrases
  if (mach->GetMType() != file_header_mtype_mseq)
    Error("CSTM: sequential machine needed\n");
  MachSeq *mseq=(MachSeq*) mach;
  if (mseq->MachGetNb()<2)
    Error("CSTM: the number of machines is suspiciously small");

    // check input layer
  if (mseq->MachGet(0)->GetMType() != file_header_mtype_mpar)
    Error("CSTM: the input layer has the wrong architecture\n");
  MachPar *mpar = (MachPar*) mseq->MachGet(0);
  if (mpar->MachGet(0)->GetMType() != file_header_mtype_tab)
    Error("CSTM: the input layer has the wrong architecture\n");
  MachTab *mtab = (MachTab*) mpar->MachGet(0);
  max_inp_idx = mtab->GetMaxInpVal();

    // check output layer
  if (mseq->MachGet(mseq->MachGetNb()-1)->GetMType() != file_header_mtype_msplit)
    Error("CSTM: the output layer has the wrong architecture\n");
  MachSplit *msp = (MachSplit*) mseq->MachGet(mseq->MachGetNb()-1);
  tg_nbphr=msp->MachGetNb();
  if (data_train && (data_train->GetOdim() != tg_nbphr)) {
    ErrorN("CSTM: output dimension of the training data should be %d, found %d\n", tg_nbphr, data_train->GetOdim());
  }

  cout << " - using cross entropy for each output vector" << endl;
  phrase_mach.clear();
  mach_errfct.clear();
  for (int m=0; m<tg_nbphr; m++) {
    phrase_mach.push_back(msp->MachGet(m));
    if (m>0 && phrase_mach[m-1]->GetOdim() != phrase_mach[m]->GetOdim())
      Error("CSTM: the output layer dimension must be identical for all phrases\n");
    //ErrFctSoftmCrossEntNgram *tmp=dynamic_cast<ErrFctSoftmCrossEntNgram*>(errfct);
    //mach_errfct.push_back(new ErrFctSoftmCrossEntNgram(*tmp));	// create copy of user specified error function
    mach_errfct.push_back(new ErrFctSoftmCrossEntNgram(*phrase_mach[m]));	// each machine gets its own error function with local mem for grad
#ifdef BLAS_CUDA
    Gpu::SetConfig(mach_errfct[m]->GetGpuConfig());
    gpu_target.push_back(Gpu::Alloc(bsize*sizeof(REAL), "targets in Trainer"));
#endif
  }
  dim_per_phrase = phrase_mach[0]->GetOdim();
  cout << " - this machine can predict up to " << phrase_mach.size() << " phrases, each with an output layer of dimension " << dim_per_phrase << endl;
  tg_slist_len = dim_per_phrase-1;


    // get source word list
  if (sr_wlist == NULL) {
    vector<WordList> *vect_wlist = NULL;
    if (data_dev != NULL)
      vect_wlist = data_dev->GetSrcWList();
    else if (data_train != NULL)
      vect_wlist = data_train->GetSrcWList();
    if ((vect_wlist != NULL) && !vect_wlist->empty())
      sr_wlist = &(vect_wlist->front());
  }
  if (sr_wlist == NULL)
    Error("no source word list available");
  if ((int) sr_wlist->GetSize() > max_inp_idx)
    ErrorN("the size of the source word list (%d) exceeds the number of input words the machine was trained for (%d)",(int) sr_wlist->GetSize(),max_inp_idx);
  debug1("* using source word list with %d words\n",(int)sr_wlist->GetSize());

    // get target word list
  if (tg_wlist == NULL) {
    vector<WordList> *vect_wlist = NULL;
    if (data_dev != NULL)
      vect_wlist = data_dev->GetTgtWList();
    else if (data_train != NULL)
      vect_wlist = data_train->GetTgtWList();
    if ((vect_wlist != NULL) && !vect_wlist->empty())
      tg_wlist = &(vect_wlist->front());
  }
  if (tg_wlist == NULL)
    Error("no target word list available");
  if (!tg_wlist->FrequSort())
    Error("the target word list doesn't contain word counts");
  if (tg_wlist->GetSize() <= tg_slist_len)
    Error("TrainerPhraseSlist: the output layer is larger than the target word list");
  debug1("* using target word list with %d words\n",(int)tg_wlist->GetSize());

  ulong sum_sl=0, sum=0;
  tg_wlist->SetShortListLength(tg_slist_len);
  tg_wlist->CountWords(sum_sl, sum);
  printf (" - setting up target short list of %d words, coverage of %5.2f%%\n", tg_slist_len, 100.0*sum_sl/sum);

#ifdef DEBUG2
  cout << "Words in slist:" << endl;
  WordID ci=tg_slist_len;
  WordList::const_iterator iter, end = tg_wlist->End();
  for (iter=tg_wlist->Begin(); (iter!=end) && (ci > 0); iter++, ci--)
    printf (" %s cnt=%d idx=%d\n", iter->word, iter->n, iter->id);
#endif

#ifdef DEBUG2
  cout << "Words not in slist:" << endl;
  for (; iter!=end; iter++)
    printf (" %s cnt=%d idx=%d\n", iter->word, iter->n, iter->id);
#endif

#ifdef DEBUG2
   // just needed for debugging
  words.reserve(tg_wlist->GetSize());
  for (iter=tg_wlist->Begin(); iter!=end; iter++) words[iter->id] = strdup(iter->word);
#endif
  
  debug0(" + done init TrainerPhraseSlist\n");
}

//
// constructor for training
//

TrainerPhraseSlist::TrainerPhraseSlist (Mach *pmach, Lrate *lrate, ErrFct *perrfct,
	const char *train_fname, const char *dev_fname, const char *pt_fname, int p_nscores,
	REAL p_wd, int p_maxep, int p_ep)
 : Trainer(pmach,lrate,perrfct,NULL,NULL,p_wd,p_maxep,p_ep),
   tg_nbphr(0), tg_slist_len(0), 
   sr_wlist(NULL), tg_wlist(NULL),
   ptable(NULL),
   nb_ex_slist(0), nb_ex_short_tgt(0),
   nb_forw(0)
{
  debug2("*** Constructor TrainerPhraseSlist for training idim=%d, odim=%d ***\n",idim,odim);
  cout << "Setting up CSTM training with short list" << endl;

  if (train_fname) {
    data_train = new Data(train_fname);
    if (idim != data_train->GetIdim()) {
      ErrorN("TrainerPhraseSlist: input dimension of the training data (%d) does not match the one of the machine (%d)\n", data_train->GetIdim(), idim);
    }
    if (data_train->GetOdim()<1 || data_train->GetOdim()>32) {
      ErrorN("TrainerPhraseSlist: output dimension of the training data should be 1..10, found %d\n", data_train->GetOdim());
    }
    auxdim = data_train->GetAuxdim();
  }
  else 
    data_train=NULL;

  if (dev_fname) {
    data_dev = new Data(dev_fname);
    data_dev_alloc=true;
    if (idim != data_dev->GetIdim()) {
      ErrorN("TrainerPhraseSlist: input dimension of the validation data (%d) does not match the one of the machine (%d)\n", data_dev->GetIdim(), idim);
    }
    if (data_dev->GetOdim()<1 || data_dev->GetOdim()>32) {
      ErrorN("TrainerPhraseSlist: output dimension of the validation data should be 1..10, found %d\n", data_dev->GetOdim());
    }
    int auxdim_dev = data_dev->GetAuxdim();
    if (0 >= auxdim)
      auxdim = auxdim_dev;
    else if (auxdim != auxdim_dev)
      ErrorN("TrainerPhraseSlist: auxiliary data dimension of the validation data should be %d, found %d", auxdim, auxdim_dev);
  }
  else {
    data_dev=NULL;
    data_dev_alloc=false;
  }
  iaux = (idim - auxdim);

  DoConstructorWork();

  if (data_dev) {
    if (pt_fname) {
      ptable = new(PtableMosesPtree);
      ptable->Read(pt_fname,5,"1:2");
    }
    else
      cout << " - no external phrase table provided (unhandled phrase pairs receive 0 logproba)" << endl;
  }
}

//
// constructor for testing
//

TrainerPhraseSlist::TrainerPhraseSlist (Mach *pmach, ErrFct *perrfct,
	Data *data, char *pt_fname, int p_nscores)
 : Trainer(pmach,NULL,perrfct,NULL,NULL),
   tg_nbphr(0), tg_slist_len(0), 
   sr_wlist(NULL), tg_wlist(NULL),
   ptable(NULL),
   nb_ex_slist(0), nb_ex_short_tgt(0),
   nb_forw(0)
{
  debug0("*** Constructor TrainerPhraseSlist for testing ***\n");
  cout << "Setting up testing with short list" << endl;

  data_train=NULL;
  data_dev=data;
  data_dev_alloc=false; // do not free it by this class !

  if (idim != data_dev->GetIdim()) {
    ErrorN("TrainerPhraseSlist: input dimension of the test data (%d) does not match the one of the machine (%d)\n", data_dev->GetIdim(), idim);
  }
  auxdim = data_dev->GetAuxdim();
  iaux = (idim - auxdim);

  DoConstructorWork();

  if (pt_fname) {
    ptable = new(PtableMosesPtree);
#ifdef BACKWARD_TM
    ptable->Read(pt_fname,5,"1:0"); // backward TM prob
#else
    ptable->Read(pt_fname,5,"1:2"); // forward TM prob
#endif
  }
  else
    cout << " - no external phrase table provided (unhandled phrase pairs receive 0 logproba)" << endl;
}

//
// constructor for nbest rescoring
//

TrainerPhraseSlist::TrainerPhraseSlist (Mach *pmach,
    WordList *p_sr_wlist, WordList *p_tg_wlist,
	char *pt_fname, int nscores, char *scores_specif)
 : Trainer(pmach,NULL,NULL,NULL,NULL),
   tg_nbphr(0), tg_slist_len(0), 
   sr_wlist(p_sr_wlist), tg_wlist(p_tg_wlist),
   ptable(NULL),
   nb_ex_short_tgt(0), nb_forw(0)
{
  debug0("*** Constructor TrainerPhraseSlist for block operations ***\n");
  cout << "Setting up CSTM with short list" << endl;
  // TODO: init with TrainerNgram before
  data_train=NULL;
  data_dev=NULL;
  DoConstructorWork();

  if (pt_fname) {
    ptable = new(PtableMosesPtree);
    ptable->Read(pt_fname, nscores, scores_specif);
  }
  else
    cout << " - no external phrase table provided (unhandled phrase pairs receive 0 logproba)" << endl;
}

//**************************************************************************************

TrainerPhraseSlist::~TrainerPhraseSlist ()
{ 
  debug0("*** Destructor TrainerPhraseSlist ***\n");

  if (buf_target_wid) delete [] buf_target_wid;
  if (buf_target_ext) delete [] buf_target_ext;
  if (buf_target_in_blocks) delete [] buf_target_in_blocks;
    // buf_input and buf_target will be deleted by ~Trainer()

#ifdef BLAS_CUDA
    // free local gpu_target buffer on each GPU
  for (vector<REAL*>::iterator it=gpu_target.begin(); it!=gpu_target.end(); ++it)
    if (*it) cudaFree(*it);
  gpu_target.clear();
#endif

  phrase_mach.clear();
  mach_errfct.clear();

#ifdef DEBUG2
  vector<char*>::const_iterator iter, end = words.end();
  for (iter=words.begin(); iter!=end; iter++) delete *iter;
  words.clear();
#endif
}


//**************************************************************************************
//
// We have MachSplit() at the ouput
// this means that each machine has its own error function with its own gradient
//   these error functions point to the outputs in the individual machines
//   and the gradients stored in this Trainer

REAL TrainerPhraseSlist::Train()
{
  if (!data_train) return -1;
#ifdef DEBUG
  printf("*****************\n");
  printf("TrainerPhraseSlist::Train():\n");
  printf(" - idim=%d, odim=%d, tg_nbphr=%d\n", idim, odim, tg_nbphr);
  printf(" -          data_in: %p \n", (void*) buf_input);
  printf(" -           target: %p \n", (void*) buf_target);
  printf(" - target_in_blocks: %p \n", (void*) buf_target_in_blocks);
  printf(" -          tgt WID: %p \n", (void*) buf_target_wid);
#endif

  Timer ttrain;		// total training time
  //Timer tload;		// total time to select examples
  //Timer ttransfer;      // total transfer time of data to GPU
  //Timer tforw;          // total forw time
  //Timer tgrad;          // total time fr gradient
  //Timer tbackw;         // total backw time

  ttrain.start();
  data_train->Rewind();

  REAL log_sum=0;
  int i;
  nb_ex=nb_ex_slist=nb_ex_short_inp=nb_ex_short_tgt=0;
  nb_tg_words=nb_tg_words_slist=0;


    // set input 
#ifdef BLAS_CUDA
  Gpu::SetConfig(mach->GetGpuConfig());
  mach->SetDataIn(gpu_input);		// we copy from buf_input to gpu_input
  debug1(" - gpu_input %p\n", gpu_input);
#else
  mach->SetDataIn(buf_input);
  debug1(" - buf_input %p\n", buf_input);
#endif

    // connect the error functions for each individual machine
    // buf_target does sequentially contain all the targets for block0, than block1 and so on
    // buf_target_in_blocks
    //  targets are arranged by blocks of bsize, i.e. first bsize targets for 1st machine, than 2nd and so on
    //  by these means we don't need to copy or re-arrange data later in the GPU
#ifdef BLAS_CUDA
  REAL *tptr;
#else
  REAL *tptr=buf_target_in_blocks;
#endif
  debug0("Error functions of the individual machines:\n");
  for (i=0; i<tg_nbphr; i++) {
    mach_errfct[i]->SetOutput(phrase_mach[i]->GetDataOut());
#ifdef BLAS_CUDA
    tptr=gpu_target[i];	// we copy later from buf_target_in_blocks to gpu_target
#endif
    mach_errfct[i]->SetTarget(tptr);
    phrase_mach[i]->SetGradOut(mach_errfct[i]->GetGrad());
    debug5(" %d: fct=%p, output=%p, target=%p, grad=%p\n",i,(void*)mach_errfct[i],(void*)phrase_mach[i]->GetDataOut(),(void*)tptr,(void*)mach_errfct[i]->GetGrad());
#ifndef BLAS_CUDA
    tptr += bsize;	// each example provides 1 target for each output machine (the word ID)
#endif
  }

  eos_src = eos_tgt = NULL_WORD;
  if (sr_wlist->HasEOS()) {
    eos_src=sr_wlist->GetEOSIndex();
    printf(" - using a special token for short source sequences (%d)\n", eos_src);
  }
  if (tg_wlist->HasEOS()) {
    eos_tgt=tg_wlist->GetEOSIndex();
    printf(" - using a special token for short target sequences (%d)\n", eos_tgt);
  }

    // master loop on all training data
  bool data_available;
  do {
    //tload.start();

      // get a bunch of data and map all the words
    int n=0;
    data_available = true;
    while (n < mach->GetBsize() && data_available) {
      data_available = data_train->Next();
      if (!data_available) break;
      debug0("TRAIN DATA: input: ");
      bool at_least_one_short=false;
      for (i=0; i<iaux; i++) { // copy word indexes
        WordID inp=(WordID) data_train->input[i];
        debug2(" %s[%d]", sr_wlist->GetWordInfo(inp).word,inp);
#if TRAINER_PHASE_SLIST_MAP_INPUT // default is not to do so
        if (inp == NULL_WORD)
          at_least_one_short=true;
        else {
          buf_input[n*idim + i] = (REAL) sr_wlist->MapIndex(inp, "TrainerPhraseSlist::Train(): input");       // map context words IDs
          if (inp==eos_src) at_least_one_short=true;
        }
#else
        buf_input[n*idim + i] = inp;
        if (inp == NULL_WORD || inp==eos_src)
          at_least_one_short=true;
        else if (inp<0 || inp>=(int)sr_wlist->GetSize())
          ErrorN("TrainerPhraseSlist::Train(): input out of bounds (%d), must be in [0,%d[", inp, (int) sr_wlist->GetSize());
#endif
      }
      for (; i < idim ; i++) // copy auxiliary data
        buf_input[n * idim + i] = data_train->input[i];
      if (at_least_one_short) nb_ex_short_inp++;

      debug0("\n - > mapped output: ");
      
      bool all_in_slist=true;  // ALL to be predicted words are in short list
      at_least_one_short=false;
      int nbtgsl=0;
      for (i=0; i<tg_nbphr; i++) {
        WordID outp=(WordID) data_train->target[i];
        int idx=i+n*tg_nbphr;
        buf_target_wid[idx] = tg_wlist->MapIndex(outp, "TrainerPhraseSlist::Train(): output");  // TODO: not really needed during training, just the current value
        if (outp==NULL_WORD
            || (at_least_one_short && outp==eos_tgt))	// we only predict the FIRST EOS, the other ones are set to NULL_WORD
        {   // NULL_WORDS are mapped, they will be detected in gradient calculation
          buf_target[idx] = (REAL) NULL_WORD;
          at_least_one_short=true;
          debug1(" -[%d->NULL]",(int) buf_target[idx]);
        }
        else {
	    // map normal word or EOS
          nb_tg_words++; // also count EOS since we need to predict them at the output
          if (outp==eos_tgt) at_least_one_short=true;
          if (tg_wlist->InShortList(buf_target_wid[idx])) {
            buf_target[idx] = (REAL) buf_target_wid[idx];
            debug3(" %s[%d->%d]", tg_wlist->GetWordInfo(outp).word,outp,(int) buf_target_wid[idx]);
            nbtgsl++;
          }
          else {
	    buf_target[idx] = (REAL) tg_slist_len;	// words that are not in slist are ALL done by the last output neuron
            debug3(" %s[%d->%d]*", tg_wlist->GetWordInfo(outp).word,outp,(int) buf_target_wid[idx]);
            all_in_slist=false;
          }
        }
      }
      if (all_in_slist) {
        nb_ex_slist++;
        nb_tg_words_slist += nbtgsl;
      }
      if (at_least_one_short) nb_ex_short_tgt++;
      debug1("     all_slist=%d\n",all_in_slist);

      n++;
    }  // loop to get a bunch of examples
    debug4("train bunch of %d words, totl=%d, totl slist=%d [%.2f%%]\n", n, nb_ex+n, nb_ex_slist, 100.0*nb_ex_slist/(nb_ex+n));
    //tload.stop();

#ifdef DEBUG2
    printf("network data:\n");
    REAL *iptr=buf_input;
    for (int nn=0;nn<n;nn++) {
       for (i=0;i<idim;i++) printf(" %f", *iptr++); printf(" -> ");
       for (i=0;i<tg_nbphr;i++) printf(" %f", *tptr++); printf("\n");
    }
#endif

      // process the bunch by the neural network
      // TODO: a lot of this code is identical with testing -> factor
    if (n>0) {
        // copy targets from buf_target to buf_target_in_blocks by re-arranging them into blocks per machine
      
      debug0("re-arrange targets\n");
      for (i=0; i<tg_nbphr; i++) {
        tptr=buf_target_in_blocks + i*bsize;	// destination start is always at full bsize blocks
        debug2(" %d starts at %p\n",i,(void*)tptr);
        REAL *tptr_src=buf_target+i;
        for (int b=0; b<n; b++) {	// be careful with bsize and current n !
          *tptr++=*tptr_src;
          tptr_src+=tg_nbphr;
        }
      }
   
#ifdef BLAS_CUDA
      //ttransfer.start();
      Gpu::MemcpyAsync(gpu_input, buf_input , n*idim*sizeof(REAL), cudaMemcpyHostToDevice);
      REAL *tptr=buf_target_in_blocks;
      for (i=0; i<tg_nbphr; i++) {
        Gpu::MemcpyAsync(gpu_target[i], tptr , n*sizeof(REAL), cudaMemcpyHostToDevice);
        tptr += n;
      }
      Gpu::StreamSynchronize();
      //ttransfer.stop();
#endif

      //tforw.start();
      mach->Forw(n,true);
      //tforw.stop();

      //tgrad.start();
      debug0("call Error functions of the individual machines:\n");
      for (i=0; i<tg_nbphr; i++) {
        debug2(" %d: %p\n",i,(void*)mach_errfct[i]);
#ifdef BLAS_CUDA
        debug2("#### CUDA: calc gradient for output %d on GPU %d\n", i, Gpu::GetCudaDevice(Gpu::GetDevice(mach_errfct[i]->GetGpuConfig())));
#endif
          // the returned log_sum is cumulated over a full batch for one specific output word
        log_sum += mach_errfct[i]->CalcGradNull(n);
      }
      //tgrad.stop();

      debug1("  log_sum=%e\n",log_sum);
#ifdef DEBUG2
      int t=(int) data_train->target[0];
# ifdef BLAS_CUDA
      Gpu::SetConfig(mach->GetGpuConfig());
      REAL * tmp = Gpu::Alloc(5, "tmp buffer for DEBUG2");
      cublasGetVector(odim,CUDA_SIZE,mach->GetDataOut(),1,tmp,1);
      printf("OUTPUT:");
      for (int i=t-2;i<=t+2; i++) printf(" %f",tmp[i]); printf("\n");
      cublasGetVector(3, CUDA_SIZE, data_train->target, 1, tmp, 1);
      printf("TARGET:");
      for (int i=0;i<1; i++) printf(" %f", tmp[i]); printf("\n");
      //TODO check if we need odim or idim!
      // TODO: cublasGetVector(odim*bsize, CUDA_SIZE, errfct->GetGrad(), 1, tmp, 1);
      printf("  GRAD:");
      for (int i=t-2;i<=t+2; i++) printf(" %f",tmp[i]); printf("\n");
      cublasFree(tmp);
# else
      printf("OUTPUT:") ; for (int i=t-2;i<=t+2; i++) printf(" %f",mach->GetDataOut()[i]); printf("\n");
      printf("TARGET:") ; for (int i=0;i<1; i++) printf(" %f",data_train->target[i]); printf("\n");
      printf("  GRAD:") ; for (int i=t-2;i<=t+2; i++) printf(" %f",errfct->GetGrad()[i]); printf("\n");
# endif //BLAS_CUDA
#endif //DEBUG2

      lrate->UpdateLrateOnForw(mach->GetNbForw());
      //tbackw.start();
      mach->Backw(lrate->GetLrate(), wdecay, n);
      //tbackw.stop();
    }

    nb_ex += n;
  } while (data_available);
#ifdef BLAS_CUDA
  Gpu::StreamSynchronize();
#endif

  ttrain.stop();
  ttrain.disp(" - training time: ");
  //tload.disp(" including load: ");
  //ttransfer.disp(" transfer: ");
  //tforw.disp(" forw: ");
  //tgrad.disp(" grad: ");
  //tbackw.disp(" backw: ");
  printf("\n");
  
  printf(" - CSTM log_sum=%.2f%s, target words=%d, in shortlist=%d, nb_tg_words_slist=%d\n",
	log_sum, tg_wlist->HasEOS() ? " including EOS" : "", nb_tg_words, nb_ex_slist, nb_tg_words_slist);
  if (nb_tg_words>0) return exp(-log_sum / (REAL) nb_tg_words);  // when normalizing consider that all examples lead to a forward pass 

  return -1;
}

//**************************************************************************************
// 

void TrainerPhraseSlist::GetMostLikelyTranslations (ofstream &fspt, REAL *optr, int ni)
{
  int Nbest=100;

    // get input length
  int input_length;
  for (input_length=0;input_length<iaux;input_length++) {
    if (buf_input[ni*idim+input_length] == NULL_WORD) break;
  }

  std::vector<std::vector<std::pair<float, std::size_t> > > prepared_scores
   = prepare_hypotheses(optr, tg_nbphr, dim_per_phrase, Nbest);
  std::vector<std::pair<float, std::vector<std::size_t> > > best
   = sort_ngrams(prepared_scores, input_length, Nbest);

  for(std::size_t i = 0; i < best.size(); ++i) {
      // source
    for (int j=0; j<iaux; j++) {
      if (buf_input[ni*idim+j] == NULL_WORD) break;
      fspt << sr_wlist->GetWordInfo(buf_input[ni*idim+j]).word << " ";
    }

      // target
    fspt << "|||";
    for(std::size_t j = 0; j < best[i].second.size(); ++j) {
      fspt << " " << tg_wlist->GetWordInfoMapped(best[i].second[j]).word;
    }

      // score
    fspt << " ||| " << exp(best[i].first);
    fspt << "\n";
  }

}

//**************************************************************************************
// 
#if 0
void TrainerPhraseSlist::GetMostLikelyTranslations (ofstream &fspt, REAL *optr, int ni)
{
  int i;
	  // Find most likely outputs
        for (i=0;i<iaux;i++) {
          if (buf_input[ni*idim+i] == NULL_WORD) break;
          fspt << sr_wlist->GetWordInfo(buf_input[ni*idim+i]).word << " ";
        }
        fspt << "||| ";
        
        for (i=0; i<tg_nbphr; i++) {
          if (buf_target_wid[i+ni*tg_nbphr] == NULL_WORD) break;
  tgrad.disp(" including ");
  tgrad.disp(" including ");
	    // find max of current word
	  REAL *sptr=optr+i*dim_per_phrase, max=*sptr++; int max_idx=0;
          for (int s=1; s<dim_per_phrase; s++, sptr++) {
            if (*sptr>max) { max=*sptr; max_idx=s; }
          }
          fspt << tg_wlist->GetWordInfoMapped(max_idx).word << "[" << max << "] ";
        }
  fspt << endl;
}
#endif
 
//**************************************************************************************
// 

REAL TrainerPhraseSlist::TestDev(char *fname)
{
  if (!data_dev) return -1;

  vector<string> src_phrase;	// interface with classical phrase tables
  vector<string> tgt_phrase;
  vector<bool> done_by_cstm;

  ofstream fs;
  if (fname) {
    cout << " - dumping phrase probability stream to file '" << fname << "'" << endl;
    fs.open(fname,ios::out);
    CHECK_FILE(fs,fname);
  }

#undef DUMP_PHRASE_TABLE
#ifdef DUMP_PHRASE_TABLE
  char *ptfname = (char*) "alltrans.txt";
  ofstream fspt;
  fspt.open(ptfname,ios::out);
  CHECK_FILE(fspt,ptfname);
  cout << " - dumping new phrase table to file '" << ptfname << "'" << endl;
#endif

  nb_ex=nb_ex_slist=nb_ex_short_inp=nb_ex_short_tgt=0;
  nb_tg_words=nb_tg_words_slist=0;
  int nb_not_in_ptable=0;	// this counts the number of phrase pairs which were not found in the external phrase table
  int nb_src_words=0;
  REAL log_sum=0;
  REAL log_sum_notunk=0;	// all known phrase pairs, either CSTM or ptable (count=nb+_ex - nb_not_in_ptable)
  REAL log_sum_cstm=0;		// only CSLM, i.e. considering phrases done by CSTM
  REAL log_sum_cstm_short=0;	// like CSTM, limited to short n-grams, i.e. we do not count the prediction of (multiple) EOS

  uint idx;

    // set input 
#ifdef BLAS_CUDA
  Gpu::SetConfig(mach->GetGpuConfig());
  mach->SetDataIn(gpu_input);		// we copy from buf_input to gpu_input
  debug1(" - gpu_input %p\n", gpu_input);
#else
  mach->SetDataIn(buf_input);
  debug1(" - buf_input %p\n", buf_input);
#endif

    // connect the error functions for each individual machine
    // buf_target does sequentially contain all the targets for block0, than block1 and so on
    // buf_target_in_blocks
    //  targets are arranged by blocks of bsize, i.e. first bsize targets for 1st machine, than 2nd and so on
    //  by these means we don't need to copy or re-arange data later in the GPU
#ifdef BLAS_CUDA
  REAL *tptr;
#else
  REAL *tptr=buf_target_in_blocks;
#endif
  debug0("Error functions of the individual machines:\n");
  for (int i=0; i<tg_nbphr; i++) {
    mach_errfct[i]->SetOutput(phrase_mach[i]->GetDataOut());
#ifdef BLAS_CUDA
    tptr=gpu_target[i];	// we copy later from buf_target_in_blocks to gpu_target
#endif
    mach_errfct[i]->SetTarget(tptr);
    phrase_mach[i]->SetGradOut(mach_errfct[i]->GetGrad());
    debug5(" %d: fct=%p, output=%p, target=%p, grad=%p\n",i,(void*)mach_errfct[i],(void*)phrase_mach[i]->GetDataOut(),(void*)tptr,(void*)mach_errfct[i]->GetGrad());
#ifndef BLAS_CUDA
    tptr += bsize;	// each example provides 1 target for each output machine (the word ID)
#endif
  }

    // how do we handle short sequences ?
  eos_src = eos_tgt = NULL_WORD;
  if (sr_wlist->HasEOS()) {
    eos_src=sr_wlist->GetEOSIndex();
    printf(" - using a special token for short source sequences (%d)\n", eos_src);
  }
  if (tg_wlist->HasEOS()) {
    eos_tgt=tg_wlist->GetEOSIndex();
    printf(" - using a special token for short target sequences (%d)\n", eos_tgt);
  }

  bool data_available;
  data_dev->Rewind();
  do {
      // get a bunch of data
    int n=0, i;
    data_available = true;
    debug0("start bunch\n");
    done_by_cstm.clear();
    while (n < mach->GetBsize() && data_available) {
      data_available = data_dev->Next();
      if (!data_available) break;

      debug0("DEV DATA: input: ");
      bool at_least_one_short=false;
      for (i=0; i<iaux; i++) { // copy word indexes
        WordID inp=(WordID) data_dev->input[i];
        idx=n*idim + i;
        debug2(" %s[%d]", tg_wlist->GetWordInfo(inp).word,inp);
#if TRAINER_PHASE_SLIST_MAP_INPUT // default is not to do so
        if (inp == NULL_WORD)
          at_least_one_short=true;
        else {
          buf_input[idx] = (REAL) sr_wlist->MapIndex(inp, "TrainerPhraseSlist::TesDev(): input");       // map context words IDs
          nb_src_words++;
          if (inp==eos_src) at_least_one_short=true;
        }
#else
        buf_input[idx] = inp;
        if (inp == NULL_WORD || inp==eos_src)
          at_least_one_short=true;
        else {
          if (inp<0 || inp>=(int)sr_wlist->GetSize())
            ErrorN("TrainerPhraseSlist::TestDev(): input out of bounds (%d), must be in [0,%d[", inp, (int) sr_wlist->GetSize());
          nb_src_words++;
        }
#endif
      }
      for (; i < idim ; i++) // copy auxiliary data
        buf_input[n * idim + i] = data_dev->input[i];
      if (at_least_one_short) nb_ex_short_inp++;

      debug0("\n - > mapped output: ");
      
      bool all_in_slist=true;  // ALL to be predicted words are in short list
      int nbtgsl=0;
      at_least_one_short=false;
      for (i=0; i<tg_nbphr; i++) {
        WordID outp=(WordID) data_dev->target[i];
        idx=i+n*tg_nbphr;
        buf_target_wid[idx] = tg_wlist->MapIndex(outp, "TrainerPhraseSlist::TestDev(): output");
        buf_target_ext[idx] = outp;		// keep unmapped target word ID for Moses phrase-table
        if (outp==NULL_WORD
            || (at_least_one_short && outp==eos_tgt))   // we only predict the FIRST EOS, the other ones are set to NULL_WORD
        {   // NULL_WORDS are mapped, they will be detected in gradient calculation
          buf_target_wid[idx] = NULL_WORD;
          buf_target[idx] = (REAL) NULL_WORD;
          at_least_one_short=true;
          debug1(" -[%d->NULL]",(int) buf_target_wid[idx]);
        }
        else {
            // map normal word or EOS
          nb_tg_words++; // also count EOS since we need to predict them at the output
          if (outp==eos_tgt) at_least_one_short=true;
          if (tg_wlist->InShortList(buf_target_wid[idx])) {
            buf_target[idx] = (REAL) buf_target_wid[idx];
            debug3(" %s[%d->%d]", tg_wlist->GetWordInfo(outp).word,outp,(int) buf_target_wid[idx]);
	    nbtgsl++;
          }
          else {
	      // TODO: we actually don't need a forward pass for words in the short lists or short n-grams
	      //       this could be used to save some time (5-10%)
            buf_target_wid[idx] = tg_slist_len;
	    buf_target[idx] = (REAL) tg_slist_len;	// words that are not in slist are ALL done by the last output neuron
            debug3(" %s[%d->%d]*", tg_wlist->GetWordInfo(outp).word,outp,(int) buf_target_wid[idx]);
            all_in_slist=false;
          }
        }
      }
      done_by_cstm.push_back(all_in_slist);
      if (all_in_slist) {
        nb_ex_slist++;
        nb_tg_words_slist += nbtgsl;
      }
      if (!at_least_one_short) nb_ex_short_tgt++;
      debug1("     all_slist=%d\n",all_in_slist);

      n++;
    }  // loop to get a bunch ef examples
    debug4("dev bunch of %d phrases, totl=%d, totl slist=%d [%.2f%%]\n", n, nb_ex+n, nb_ex_slist, 100.0*nb_ex_slist/(nb_ex+n));

#ifdef DEBUG2
printf("network data:\n");
REAL *iptr=buf_input;
REAL *tptr=buf_target;
for (int nn=0;nn<n;nn++) {
   for (i=0;i<idim;i++) printf(" %f", *iptr++); printf(" -> ");
   for (i=0;i<tg_nbphr;i++) printf(" %f", *tptr++); printf("\n");
}
#endif


      // process the bunch by the neural network
    if (n>0) {
        // copy targets from buf_target to buf_target_in_blocks by re-arranging them into blocks per machine
      
      debug0("re-arrange targets\n");
      for (i=0; i<tg_nbphr; i++) {
        tptr=buf_target_in_blocks + i*bsize;	// destination start is always at full bsize blocks
        debug2(" %d starts at %p\n",i,(void*)tptr);
        REAL *tptr_src=buf_target+i;
        for (int b=0; b<n; b++) {	// be careful with bsize and current n !
          *tptr++=*tptr_src;
          tptr_src+=tg_nbphr;
        }
      }
    
#ifdef BLAS_CUDA
      Gpu::MemcpyAsync(gpu_input, buf_input , n*idim*sizeof(REAL), cudaMemcpyHostToDevice);
      REAL *tptr=buf_target_in_blocks;
      for (i=0; i<tg_nbphr; i++) {
        Gpu::MemcpyAsync(gpu_target[i], tptr , n*sizeof(REAL), cudaMemcpyHostToDevice);
        tptr += n;
      }
      Gpu::StreamSynchronize();
#endif
      mach->Forw(n,false); 
      for (i=0; i<tg_nbphr; i++) {
          // the returned log_sum is cumulated over a full batch for one specific output word
        //log_sum += mach_errfct[i]->CalcValueNull(n);
        log_sum += mach_errfct[i]->CalcGradNull(n);	// TODO: should use CalcValueNull()
      }
    }

#if DIRECT_PROBA_CALCULATION
      // get probas from CSLM or back-off LM
#ifdef BLAS_CUDA
      // host output is of dim bsize*odim - bsize*tg_nphr*dim_per_phrase
      // it contains the whole bunch of the 1st output, then whole bunch of 2nd output, etc
    for (int i=0; i<tg_nbphr; i++) {
      Gpu::MemcpyAsync(host_output+i*bsize*dim_per_phrase,phrase_mach[i]->GetDataOut(), n*dim_per_phrase*sizeof(REAL), cudaMemcpyDeviceToHost);
      // TODO: we actually copy too much data, for each output vector we only need one value !
    }
    Gpu::StreamSynchronize();
#endif

    debug1("Collect n=%d\n", n);
    if (n!=(int) done_by_cstm.size())
      Error("TrainerPhraseSlist::TestDev(): internal error, number of phrases done by CSTM does not match");

    REAL *ptr_input = buf_input;	// n times idim values
    for (int ni=0; ni<n; ni++) {
      REAL logP=0.0, logP_short=0.0;
      if (done_by_cstm[ni]) {
          // get proba from CSTM (removed renorm)

        for (i=0; i<tg_nbphr; i++) {
          WordID cur_tg=buf_target_wid[i+ni*tg_nbphr];
          if (cur_tg == NULL_WORD) break;
		// get proba from output i for bunch ni
#ifdef BLAS_CUDA
	  REAL *optr=host_output+i*bsize*dim_per_phrase + ni*dim_per_phrase;
#else
	  REAL *optr=phrase_mach[i]->GetDataOut() + ni*dim_per_phrase;
#endif
          logP += safelog(optr[cur_tg]); // no error check on indices necessary here
          if (buf_target_ext[i+ni*tg_nbphr] != eos_tgt) { // exclude the (easy) prediction of EOS from stats
            logP_short += safelog(optr[cur_tg]); // no error check on indices necessary here
          }
          debug5("n=%3d, pos=%d, tg_w=%d (unmapped %d), P=%f\n",ni,i,cur_tg,buf_target_ext[i+ni*tg_nbphr],optr[cur_tg]);
        }
        debug4(" -      -> logP=%f/%d, logP_short=%f/%d\n",logP,logP_short); 

#ifdef DUMP_PHRASE_TABLE
          // create output phrase table
        for (i=0;i<iaux;i++) {
          if (buf_input[ni*idim+i] == NULL_WORD) break;
          fspt << sr_wlist->GetWordInfo(buf_input[ni*idim+i]).word << " ";
        }
        fspt << "||| ";
        for (i=0;i<tg_nbphr;i++) {
          if (buf_target_wid[i+ni*tg_nbphr] == eos_tgt) break;
          fspt << tg_wlist->GetWordInfoMapped(buf_target_wid[ni*tg_nbphr+i]).word << " ";
        }
        fspt << "||| " << logP << endl;
#endif

#ifdef DUMP_PHRASE_TABLE_NBEST
	Error("GetMostLikelyTranslations() change to work with multiple output vectors");
        GetMostLikelyTranslations(fspt,optr,ni);
#endif

        debug1(" CSLM: logP=%e\n", logP);
        log_sum_cstm += logP;
        log_sum_cstm_short += logP_short;
        log_sum_notunk += logP;
        log_sum += logP;
      }
      else {
Error("not done by CSTM");

       if (ptable) {
          // request proba from Moses phrase-table
         debug0("create textual phrase pair for external phrase table (word + index)\n");
         src_phrase.clear();
         debug0("  source:");
         for (i=0; i<iaux && ptr_input[i]!=NULL_WORD; i++) {
           src_phrase.push_back(sr_wlist->GetWordInfo((uint) ptr_input[i]).word);	// TODO: char* to string
           debug2(" %s[%d]", src_phrase.back().c_str(), (uint) ptr_input[i]);
#ifdef DUMP_PHRASE_TABLE
           fspt << src_phrase.back() << " ";
#endif
         }

#ifdef DUMP_PHRASE_TABLE
         fspt << "|P| ";
#endif
         tgt_phrase.clear();
         debug0("  target:");
         for (i=0; i<tg_nbphr && buf_target_ext[i+ni*tg_nbphr]!=eos_tgt; i++) {
           tgt_phrase.push_back(tg_wlist->GetWordInfoMapped(buf_target_ext[i+ni*tg_nbphr]).word);	// TODO: char* to string
           debug2(" %s[%d]", tgt_phrase.back().c_str(), buf_target_ext[i+ni*tg_nbphr]);
#ifdef DUMP_PHRASE_TABLE
           fspt << tgt_phrase.back() << " ";
#endif
         }
# ifdef BACKWARD_TM
         logP = ptable->GetProb(tgt_phrase, src_phrase);
# else
         logP = ptable->GetProb(src_phrase, tgt_phrase);
# endif
         if (logP == PROBA_NOT_IN_PTABLE) nb_not_in_ptable++;
                                     else log_sum_notunk += logP;
         logP = safelog(logP); // take log now
         debug1("  => logP=%e\n",logP);
         log_sum += logP;
       }
       else { // no ptable was specified
         logP=0; // flag output that it wasn't done by CSTM
       }
#ifdef DUMP_PHRASE_TABLE
       fspt << "||| " << logP << endl;
#endif
      } // not done by CSTM
          
      ptr_input += idim;  // next example in bunch at input
      if (fname) {
        fs << logP << endl;
      }
    }
#endif // old proba calculation

    nb_ex += n;
    debug2("%d: %f\n",nb_ex,exp(-log_sum/nb_ex));
  } while (data_available);

  printf(" - %d phrases, %d target words, avr length src=%.1f tgt=%.1f, CSTM: %d phrases (%.2f), %d target words (%.2f)\n",
	 nb_ex, nb_tg_words, (REAL) nb_src_words/nb_ex, (REAL) nb_tg_words/nb_ex,
	 nb_ex_slist, 100.0*nb_ex_slist/nb_ex, nb_tg_words_slist, 100.0 * nb_tg_words_slist/nb_tg_words);
  if (ptable) {
    printf(" - %d words were looked up in external phrase table, %d (%.2f%% were not found)\n",
	nb_ex-nb_ex_slist, nb_not_in_ptable, 100.0*nb_not_in_ptable/(nb_ex-nb_ex_slist));
  }

#ifdef DIRECT_PROBA_CALCULATION
  REAL px = (nb_ex>0) ? exp(-log_sum / (REAL) nb_ex) : -1;
  printf("   cstm px=%.2f, ln_sum=%.2f, cstm_short_px=%.2f, ln_sum=%.2f, overall px=%.2f, with unk=%.2f\n",
        (nb_ex_slist>0) ? exp(-log_sum_cstm / (REAL) nb_ex_slist) : -1, log_sum_cstm,
        (nb_ex_slist>0) ? exp(-log_sum_cstm_short / (REAL) nb_ex_slist) : -1, log_sum_cstm_short,
        (nb_ex-nb_not_in_ptable>0) ? exp(-log_sum_notunk / (REAL) (nb_ex-nb_not_in_ptable)) : -1,
        px);
#else
  REAL px = (nb_ex>0) ? exp(-log_sum / (REAL) nb_tg_words_slist) : -1;
  printf("   px=%.2f, ln_sum=%.2f\n", px, log_sum);
#endif

  if (fname) fs.close();
#ifdef DUMP_PHRASE_TABLE
  fspt.close();
#endif

  return px;
}


//**************************************************************************************
// information after finishing an epoch

void TrainerPhraseSlist::InfoPost ()
{
    // if EOS is predicted by the NN, we don't count it as short
  printf(" - epoch finished, %d target words in %d phrases (%.2f/%.2f%% short source/target)\n",
	nb_tg_words, nb_ex,
	100.0*nb_ex_short_inp/nb_ex, 100.0*nb_ex_short_tgt/nb_ex);
  printf("   CSTM: %d target words in %d phrases (%.2f%%), avrg px=%.2f\n",
	nb_tg_words_slist, nb_ex_slist, 100.0*nb_ex_slist/nb_ex,
	err_train);
}

//**************************************************************************************
// request one n-gram probability, usually the called will be delayed
// and processes later 


//**************************************************************************************
// collect all delayed probability requests


void TrainerPhraseSlist::ForwAndCollect(vector< vector<string> > &src_phrases, AlignReq *areq, int req_beg, int req_end, int bs, int tm_pos)
{
  if (bs<=0) return;
  debug3("TrainerPhraseSlist::ForwAndCollect(): collecting outputs %d .. %d from bunch of size %d\n", req_beg, req_end, bs);
  debug3("\ttarget machines %d x dim %d = total %d\n", tg_nbphr, dim_per_phrase, odim);

  if (bs != (int) src_phrases.size())
    ErrorN("TrainerPhraseSlist::ForwAndCollect(): the number of source phrases (%d) does not match block length (%d)", (int) src_phrases.size(), bs);

#ifdef DEBUG
  printf("bunch of %d\n",bs);
  for (int b=0; b<bs; b++) {
    printf("%3d:", b);
    for (int ii=0; ii<idim; ii++) printf(" %.2f", buf_input[b*idim+ii]); printf("\n");
  }
#endif

  nb_forw++;
#ifdef CUDA
  Gpu::SetConfig(mach->GetGpuConfig());
  mach->SetDataIn(gpu_input);
  Gpu::MemcpyAsync(gpu_input, buf_input , bs*idim*sizeof(REAL), cudaMemcpyHostToDevice);
#else
  mach->SetDataIn(buf_input);
#endif
  mach->Forw(bs,false);

#ifdef BLAS_CUDA
  for (int tw=0; tw<tg_nbphr; tw++)
    Gpu::MemcpyAsync(host_output + tw*bsize*dim_per_phrase, phrase_mach[tw]->GetDataOut(), bs*dim_per_phrase*sizeof(REAL), cudaMemcpyDeviceToHost);
  Gpu::StreamSynchronize();
#endif

    // stats
  int cnt_ex_slist=0, cnt_tg_words=0, cnt_tg_words_slist=0;

  for (int n=req_beg; n<=req_end; n++) {
    REAL logP=0;
    int b=areq[n].bs;

    if ((int) areq[n].tgph.size() > tg_nbphr)
      ErrorN("TrainerPhraseSlist::ForwAndCollect(): target phrase too long (%d) for machine (%d)", (int) areq[n].tgph.size(), tg_nbphr);

#ifdef DEBUG
    printf("collect b=%3d \n input:", b);
    for (int ii=0; ii<idim; ii++) printf(" %f",buf_input[b*idim+ii]); printf("\n");
#endif

      // map target words
    debug0(" output:");
    bool all_in_slist=true;
    int tw;
    for (tw=0; all_in_slist && tw<tg_nbphr; tw++) {
      WordID outp = areq[n].tgwid[tw];
      debug1(" %d",outp);
      if (outp==eos_tgt) break;
      cnt_tg_words++;
      buf_target_wid[tw] = tg_wlist->MapIndex(outp, "TrainerPhraseSlist::ForwAndCollect() output");
      debug1("->%d",buf_target_wid[tw]);
      all_in_slist=tg_wlist->InShortList(buf_target_wid[tw]);
    }
      // fill up
    for (; tw<tg_nbphr; tw++) {
      debug0(" fill");
      buf_target_wid[tw]=eos_tgt;
    }
    debug1("    slist=%d\n",all_in_slist);

    if (!all_in_slist) {
        // get proba from external phrase table
      logP=safelog(ptable->GetProb(src_phrases[areq[n].bs], areq[n].tgph));
      debug1(" ptable: logP=%f\n", logP);
    }
    else {
        // get proba from CSLM
      debug0(" -  in slist CSLM:");
      logP=0; int cnt=0;
      for (int tw=0; tw<tg_nbphr; tw++) {
        if (buf_target_wid[tw] == eos_tgt) break;
#ifdef BLAS_CUDA
        //old;  REAL *optr=host_output + b*odim;
        //test: REAL *optr=host_output+i*bsize*dim_per_phrase + ni*dim_per_phrase;
        REAL *optr=host_output+tw*bsize*dim_per_phrase + b*dim_per_phrase;
#else
        //old: REAL *optr=mach->GetDataOut() + b*odim;
        //test: REAL *optr=phrase_mach[i]->GetDataOut() + ni*dim_per_phrase;
        //TODO: it would be much more efficient to do all the examples of one machine and then switch to the next one
        REAL *optr=phrase_mach[tw]->GetDataOut() + b*dim_per_phrase;
#endif
        debug1(" %e", optr[buf_target_wid[tw]]);
        logP += safelog(optr[buf_target_wid[tw]]);
        cnt++;
      }
      if (cnt==0) Error("no target phrases when collecting output");
      logP /= cnt; // TODO: is this normalization correct ?
      debug1(" -> log avr=%f\n",logP);

      cnt_ex_slist++;
      cnt_tg_words_slist += cnt;
    }

        // store LM proba
    areq[n].hyp->AddFeature(logP,tm_pos);
  } // for (ni=...)

  printf(" nb of phrases: %d with %d target words, by CSTM %d (%5.2f%%), avrg length %1.2f words\n",
	 req_end-req_beg+1, cnt_tg_words, cnt_ex_slist, (float) 100.0* cnt_ex_slist / (req_end-req_beg+1), (float) cnt_tg_words_slist/cnt_ex_slist);
  nb_ex += (req_end-req_beg+1);
  nb_ex_slist += cnt_ex_slist;
  nb_tg_words_slist += cnt_tg_words_slist;
  nb_tg_words += cnt_tg_words;
}


void TrainerPhraseSlist::BlockStats() {
   //printf(" - %d phrase probability requests, %d=%5.2f short phrase %d forward passes (avrg of %d probas), %d=%5.2f%% predicted by CSTM\n",
	//nb_ngram, nb_ex_short_tgt, 100.0*nb_ex_short_tgt/nb_ngram, nb_forw, nb_ngram/nb_forw, nb_ex_slist, 100.0*nb_ex_slist/nb_ngram);
   printf(" - CSTM: %d forward passes, %d=%5.2f%% phrases were predicted by CSTM\n",
	nb_forw, nb_ex_slist, 100.0 * nb_ex_slist/nb_ex);
}
