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
#include "TrainerNgramSlist.h"

#ifdef LM_KEN
#include "BackoffLmKen.h"
#endif
#ifdef LM_SRI
#include "BackoffLmSri.h"
#endif

#define CSLM_DOES_SHORT_NGRAMS

void TrainerNgramSlist::DoConstructorWork() {
    // check machine type
  if (mach->GetMType() != file_header_mtype_mseq)
    Error("CSLM: sequential machine needed\n");
  MachSeq *mseq=(MachSeq*) mach;

    // check input layer
  // TODO: More robust detection when there are multiple nested MachSeq,
  // MachPar, or MachJoin
  if (mseq->MachGet(0)->GetMType() != file_header_mtype_mpar)
    Error("CSLM: the input layer has the wrong architecture\n");
  MachPar *mpar = (MachPar*) mseq->MachGet(0);
  if (mpar->MachGet(0)->GetMType() != file_header_mtype_tab)
    Error("CSLM: the input layer has the wrong architecture\n");
  MachTab *mtab = (MachTab*) mpar->MachGet(0);
  max_inp_idx = mtab->GetMaxInpVal();

    // get word list and target position
  tgpos = iaux;
  if (wlist == NULL) {
    vector<WordList> *vect_wlist = NULL;
    if (data_dev != NULL) {
      vect_wlist = data_dev->GetTgtWList();
      tgpos = data_dev->GetTgPos();
    }
    else if (data_train != NULL) {
      vect_wlist = data_train->GetTgtWList();
      tgpos = data_train->GetTgPos();
    }
    if ((vect_wlist != NULL) && !vect_wlist->empty())
      wlist = &(vect_wlist->front());
  }

  if (wlist == NULL)
    Error("no word list available");
  if (!wlist->FrequSort())
    Error("the word list don't contain word count");
  if ((int) wlist->GetSize() > max_inp_idx)
    Error("the size of the word list exceeds the number of input words the machine was trained for");

  ulong sum_sl=0, sum=0;
  wlist->SetShortListLength(slist_len);
  wlist->CountWords(sum_sl, sum);
  printf (" - setting up short list of %d words, coverage of %5.2f%%\n", slist_len, 100.0*sum_sl/sum);

#ifdef DEBUG2
  cout << "Words in slist:" << endl;
  WordID ci=slist_len;
  WordList::const_iterator iter, end = wlist->End();
  for (iter=wlist->Begin(); (iter!=end) && (ci > 0); iter++, ci--)
    printf (" %s cnt=%d idx=%d\n", iter->word, iter->n, iter->id);
#endif

#ifdef DEBUG2
  cout << "Words not in slist:" << endl;
  for (; iter!=end; iter++)
    printf (" %s cnt=%d idx=%d\n", iter->word, iter->n, iter->id);
#endif

#ifdef DEBUG2
   // just needed for debugging
  words.reserve(wlist->GetSize());
  words.resize(wlist->GetSize());
  for (iter=wlist->Begin(); iter!=end; iter++) words[iter->id] = strdup(iter->word);
#endif

    // load back-off LM and set up vocab mapping
    // the maximum order of the back-off LM is the target position in n-gram + 1
#ifdef LM_KEN
  blm = new BackoffLmKen(lm_fname,tgpos+1,*wlist);
#endif
#ifdef LM_SRI
  blm = new BackoffLmSri(lm_fname,tgpos+1,*wlist);
#endif

  BlockSetMax();  // allocate req

}

//
//
//

TrainerNgramSlist::TrainerNgramSlist (Mach *pmach, Lrate *lrate, ErrFct *perrfct,
	const char *train_fname, const char *dev_fname, const char *p_lm_fname,
	REAL p_wd, int p_maxep, int p_ep)
 : TrainerNgram(pmach,lrate,perrfct,train_fname,dev_fname,p_wd,p_maxep,p_ep),
   nb_ex_slist(0), nb_ex_short(0),
   lm_fname(strdup(p_lm_fname)), lm_buf_target(new WordID[odim*bsize]),
   slist_len(mach->GetOdim()-1), blm(NULL), wlist(NULL), max_req(0), nreq(0), req(NULL), nb_ngram(0), nb_forw(0)
{
  cout << "Setting up training with short list" << endl;
  DoConstructorWork();
}

//
//
//

TrainerNgramSlist::TrainerNgramSlist (Mach *pmach, ErrFct *perrfct,
	Data *data, char *p_lm_fname)
 : TrainerNgram(pmach,perrfct,data),
   nb_ex_slist(0), nb_ex_short(0),
   lm_fname(strdup(p_lm_fname)), lm_buf_target(new WordID[odim*bsize]),
   slist_len(mach->GetOdim()-1), blm(NULL), wlist(NULL), max_req(0), nreq(0), req(NULL), nb_ngram(0), nb_forw(0)
{
  cout << "Setting up testing with short list" << endl;
  DoConstructorWork();
}

TrainerNgramSlist::TrainerNgramSlist (Mach *pmach, WordList *wlist, char *p_lm_fname, int aux_dim)
 : TrainerNgram(pmach,NULL,NULL, aux_dim),
   nb_ex_slist(0), nb_ex_short(0),
   lm_fname(strdup(p_lm_fname)), lm_buf_target(new WordID[odim*bsize]),
   slist_len(mach->GetOdim()-1), blm(NULL), wlist(wlist), max_req(0), nreq(0), req(NULL), nb_ngram(0), nb_forw(0)
{
  cout << "Setting up CSLM with short list" << endl;
  DoConstructorWork();
}

void TrainerNgramSlist::FreeReq()
{
  if (req) {
    for (int i=0; i<nreq; i++) {
      free(req[i].ctxt);
      if (req[i].aux) delete [] req[i].aux;
    }
  }
  nreq=0;
}
//**************************************************************************************

TrainerNgramSlist::~TrainerNgramSlist ()
{ 

  if (lm_fname) free(lm_fname);
  delete [] lm_buf_target;

  if (blm) delete blm;

#ifdef DEBUG2
  vector<char*>::const_iterator iter, end = words.end();
  for (iter=words.begin(); iter!=end; iter++) delete *iter;
  words.clear();
#endif

  FreeReq();
  if (req) delete [] req;
}


//**************************************************************************************
// special version for GPU cards that load all examples on the card 
// and than runs a whole epoch without explicit data transfer

#ifdef BLAS_CUDA_NEW
REAL TrainerNgramSlist::Train()
{
  if (!data_train) return -1;

#ifdef DEBUG
  printf("*****************\n");
  printf("TrainerNgramSlist::Train() on GPU:\n");
  printf(" -  data_in: %p \n", (void*) buf_input);
  printf(" -   target: %p \n", (void*) buf_target);
  printf(" -  tgt WID: %p \n", (void*) buf_target_wid);
  printf(" - grad_out: %p \n", (void*) errfct->GetGrad());
#endif

  Timer ttrain;		// total training time
  ttrain.start();

  int n, i;
  nb_ex=nb_ex_slist=nb_ex_short=0;

  Gpu::SetConfig(mach->GetGpuConfig());
  mach->SetDataIn(gpu_input);		// we copy from buf_input to gpu_input
  errfct->SetTarget(gpu_target);	// we copy from buf_target to gpu_target
  errfct->SetOutput(mach->GetDataOut());
  mach->SetGradOut(errfct->GetGrad());
  data_train->Rewind();
  
    // reserve memory on the GPU for all examples
  int mem_ex=data_train->GetNb();
  printf(" - allocating memory for %d examples on GPU (%.1f MBytes)\n",mem_ex,mem_ex*(idim+1)*sizeof(REAL)/1024.0/1024.0);
  REAL *gpu_input_all = Gpu::Alloc(mem_ex*idim, "all training data");
  REAL *gpu_target_all = Gpu::Alloc(mem_ex*1, "all targets");

  bool data_available;
  REAL *gpu_iptr=gpu_input_all, *gpu_tptr=gpu_target_all;
  do {
      // get a bunch of data and map all the words
    n=0;
    data_available = true;
    while (n < mach->GetBsize() && data_available) {
      data_available = data_train->Next();
      if (!data_available) break;
      buf_target_wid[n] = wlist->MapIndex((WordList::WordIndex) data_train->target[0], "TrainerNgramSlist::Train(): target");	// map target word IDs
      buf_target[n] = (REAL) buf_target_wid[n];
      if (wlist->InShortList(buf_target_wid[n]))
        nb_ex_slist++;
      else {
	buf_target[n] = (REAL) slist_len;	// words that are not in slist are ALL done by the last output neuron
        buf_target_wid[n] = slist_len;
      }

      bool at_least_one_short=false;
      for (i=0; i<iaux; i++) { // copy word indexes
         WordList::WordIndex inp=(WordList::WordIndex) data_train->input[i];
         buf_input[n*idim + i] = (REAL) wlist->MapIndex(inp, "TrainerNgramSlist::Train(): input"); // map context words IDs
         if (inp == NULL_WORD)
           at_least_one_short=true;
      }
      for (; i < idim ; i++) // copy auxiliary data
        buf_input[n * idim + i] = data_train->input[i];
      if (at_least_one_short) nb_ex_short++;
      n++;
    }

    if (nb_ex+n > mem_ex) {
      ErrorN("trying to load %d examples, but memory was reserved for %d examples only\n", nb_ex, mem_ex);
    }

    if (n>0) {
      Gpu::MemcpyAsync(gpu_iptr, buf_input , n*idim*sizeof(REAL), cudaMemcpyHostToDevice);
      Gpu::MemcpyAsync(gpu_tptr, buf_target , n*1*sizeof(REAL), cudaMemcpyHostToDevice);
      Gpu::StreamSynchronize();
      gpu_iptr += n*idim;
      gpu_tptr += n*1;
    }

    nb_ex += n;
  } while (data_available);

  printf(" - training on %d examples on GPU\n", nb_ex);

  Timer tgrad;		// profiling: time to calculate gradients
  Timer bprop;          // profiling: time to compute the bprop

  gpu_iptr = gpu_input_all;
  gpu_tptr = gpu_target_all;
  errfct->InitGradCumul();
  n=0;
  while (n<nb_ex) {
    int b = nb_ex-n;
    if (b>bsize) b=bsize;
    mach->SetDataIn(gpu_iptr);
    errfct->SetTarget(gpu_tptr);

    mach->Forw(b,true); 
    tgrad.start();
    errfct->CalcGradCumul(b);
    tgrad.stop();
    bprop.start();
    mach->Backw(lrate->GetLrate(), wdecay, b);


    n += b;
    gpu_iptr += b*idim;
    gpu_tptr += b*1;
  } 

  cublasFree (gpu_input_all); 
  cublasFree (gpu_target_all); 

  ttrain.stop();
  ttrain.disp(" - training time: ");
  tgrad.disp(" including grad: ");
  bprop.disp(", bprop: ");
  printf("\n");

  REAL log_sum = errfct->GetGradCumul();
  if (nb_ex>0) return exp(-log_sum / (REAL) nb_ex);  // return perplexity

  return -1;
}
#endif

//**************************************************************************************

#if 1
REAL TrainerNgramSlist::Train()
{
  if (!data_train) return -1;

#ifdef DEBUG
  printf("*****************\n");
  printf("TrainerNgramSlist::Train():\n");
  printf(" -  data_in: %p \n", (void*) buf_input);
  printf(" -   target: %p \n", (void*) buf_target);
  printf(" -  tgt WID: %p \n", (void*) buf_target_wid);
  printf(" - grad_out: %p \n", (void*) errfct->GetGrad());
#endif
  data_train->Rewind();
  Timer ttrain;		// total training time
  ttrain.start();
  Timer tgrad;		// profiling: time to calculate gradients
  Timer bprop;          // profiling: time to compute the bprop

  int i;
  REAL log_sum=0;
  nb_ex=nb_ex_slist=nb_ex_short=0;

#ifdef BLAS_CUDA
  Gpu::SetConfig(mach->GetGpuConfig());
  mach->SetDataIn(gpu_input);		// we copy from buf_input to gpu_input
  errfct->SetTarget(gpu_target);	// we copy from buf_target to gpu_target
#else
  mach->SetDataIn(buf_input);
  errfct->SetTarget(buf_target);
#endif
  errfct->SetOutput(mach->GetDataOut());
  mach->SetGradOut(errfct->GetGrad());

  bool data_available;
  do {
      // get a bunch of data and map all the words
    int n=0;
    data_available = true;
    while (n < mach->GetBsize() && data_available) {
      data_available = data_train->Next();
      if (!data_available) break;
      buf_target_wid[n] = wlist->MapIndex((WordList::WordIndex) data_train->target[0], "TrainerNgramSlist::Train(): target");	// map target word IDs
      buf_target[n] = (REAL) buf_target_wid[n];
      if (wlist->InShortList(buf_target_wid[n]))
        nb_ex_slist++;
      else {
	buf_target[n] = (REAL) slist_len;	// words that are not in slist are ALL done by the last output neuron
        buf_target_wid[n] = slist_len;
      }

      bool at_least_one_short=false;
      for (i=0; i<iaux; i++) { // copy word indexes
         WordList::WordIndex inp=(WordList::WordIndex) data_train->input[i];
         buf_input[n*idim + i] = (REAL) wlist->MapIndex(inp, "TrainerNgramSlist::Train(): input"); // map context words IDs
         if (inp == NULL_WORD)
           at_least_one_short=true;
      }
      for (; i < idim ; i++) // copy auxiliary data
        buf_input[n * idim + i] = data_train->input[i];
      if (at_least_one_short) nb_ex_short++;

#ifdef DEBUG
      printf("Data n=%d\n",n);
      printf("Input: "); for (i=0; i<idim; i++) { printf(" %d", (int) data_train->input[i]); int word_index=(int) data_train->input[i]; printf("\"%s\"", wlist->GetWordInfo(word_index).word); } printf(" -> %d\n", (int) data_train->target[0]);
      printf("Mapped:"); for (i=0; i<idim; i++) { printf(" %d", (int) buf_input[n*idim+i]); int word_index=(int) buf_input[n*idim+i]; printf("\"%s\"", wlist->GetWordInfoMapped(word_index).word); } printf(" -> %d\n", (int) buf_target[n]);
      printf("Aux:"); for (i=iaux; i<idim; i++) printf(" %f \n", buf_input[n*idim+i]);
#endif
      n++;
    }

    if (n>0) {
#ifdef BLAS_CUDA
      Gpu::MemcpyAsync(gpu_input, buf_input , n*idim*sizeof(REAL), cudaMemcpyHostToDevice);
      Gpu::MemcpyAsync(gpu_target, buf_target , n*1*sizeof(REAL), cudaMemcpyHostToDevice);
#endif
      mach->Forw(n,true); 
      tgrad.start();
      log_sum += errfct->CalcGrad(n);
      tgrad.stop();
      lrate->UpdateLrateOnForw(mach->GetNbForw());
      bprop.start();
      mach->Backw(lrate->GetLrate(), wdecay, n);
      bprop.stop();
    }

    nb_ex += n;
    //if (nb_ex % 10000 == 0) printf("%d ex\n", nb_ex);
  } while (data_available);
#ifdef BLAS_CUDA
  Gpu::StreamSynchronize();
#endif

  ttrain.stop();
  ttrain.disp(" - training time: ");
  tgrad.disp(" including grad: ");
  bprop.disp(", bprop: ");
  printf("\n");
  
  if (nb_ex_slist>0) return exp(-log_sum / (REAL) nb_ex_slist);  // return perplexity

  return -1;
}
#endif

//**************************************************************************************
// 

REAL TrainerNgramSlist::DoTestDev(char *fname, bool renorm)
{
  if (!data_dev) return -1;

  ofstream fs;
  if (fname) {
    cout << " - dumping log probability stream to file '" << fname << "'" << endl;
    fs.open(fname,ios::out);
    CHECK_FILE(fs,fname);
    fs.precision(8);
    fs << std::scientific;
  }

  int nb_ex=nb_ex_slist=nb_ex_short=0;
  REAL logP, log_sum=0;
  REAL log_sum_cslm=0;	// only CSLM, i.e. considering all words out of slist as one prediction
  int lm_order=blm->GetOrder();

#ifdef BLAS_CUDA
  Gpu::SetConfig(mach->GetGpuConfig());
  mach->SetDataIn(gpu_input);		// we copy from buf_input to gpu_input
  errfct->SetTarget(gpu_target);	// we copy from buf_target to gpu_target
#else
  mach->SetDataIn(buf_input);
  errfct->SetTarget(buf_target);
#endif
  errfct->SetOutput(mach->GetDataOut());

  bool data_available;
  data_dev->Rewind();
  do {
      // get a bunch of data
    int n=0, i;
    data_available = true;
    while (n < mach->GetBsize() && data_available) {
      data_available = data_dev->Next();
      if (!data_available) break;
      for (i=0; i<iaux; i++) // copy word indexes
         buf_input[n*idim + i] = (REAL) wlist->MapIndex((WordList::WordIndex) data_dev->input[i], "TrainerNgramSlist::DoTestDev(): input"); // map context words IDs
      for (; i < idim ; i++) // copy auxiliary data
        buf_input[n * idim + i] = data_dev->input[i];

      buf_target_wid[n] = wlist->MapIndex((WordList::WordIndex) data_dev->target[0], "TrainerNgramSlist::DoTestDev(): target");	// map target word IDs
      lm_buf_target[n] = buf_target_wid[n];	// keep unmapped target word ID for back-off LM
      buf_target[n] = (REAL) buf_target_wid[n];
	  // TODO: we actually don't need a forward pass for words in the short lists or short n-grams
	  //       this could be used to save some time (5-10%)
      if (!wlist->InShortList(buf_target_wid[n])) {
        buf_target_wid[n] = slist_len;		// words that are not in slist are ALL done by the last output neuron
	buf_target[n] = (REAL) slist_len;
      }

#ifdef DEBUG
      printf("Data n=%d\n",n);
      printf("  input: "); for (i=0; i<idim; i++) printf(" %6d", (int) data_dev->input[i]); printf(" -> %6d\n", (int) data_dev->target[0]);
      printf("  mapped:"); for (i=0; i<idim; i++) printf(" %6d", (int) buf_input[n*idim+i]); printf(" -> %6d\n", (int) buf_target[n]);
#endif

      n++;
    }

      // process the bunch by the neural network
    if (n>0) {
#ifdef BLAS_CUDA
      Gpu::MemcpyAsync(gpu_input, buf_input , n*idim*sizeof(REAL), cudaMemcpyHostToDevice);
      Gpu::MemcpyAsync(gpu_target, buf_target , n*1*sizeof(REAL), cudaMemcpyHostToDevice);
#endif
      mach->Forw(n,false); 
      log_sum_cslm += errfct->CalcValue(n);
    }

      // get probas from CSLM or back-off LM
#ifdef BLAS_CUDA
    cudaMemcpy(host_output, mach->GetDataOut(), n*odim*sizeof(REAL), cudaMemcpyDeviceToHost);
    REAL *optr=host_output;
#else
    REAL *optr=mach->GetDataOut();
#endif
    REAL *ptr_input = buf_input;
    for (int ni=0; ni<n; ni++) {

#ifdef DEBUG
      printf("n=%d: predict", ni);
      for (int ii=0; ii<idim; ii++) printf(" %d", (WordID) ptr_input[ii]);
      printf(" -> %d   ", lm_buf_target[ni]);
#endif 
        // if the current n-gram has a NULL_WORD in the first place -> find the shortest n-gram order and request it
        // DataNext() will take care to propose the next longer n-gram
#ifndef CSLM_DOES_SHORT_NGRAMS
      if ((WordID) ptr_input[0] == NULL_WORD) {
        int p;
        for (p=tgpos-2; p>=0 && (ptr_input[p]!=NULL_WORD); p--);
        //logP = blm->BoffLnPid(ptr_input+max(tgpos+1-lm_order, 0), lm_buf_target[ni], tgpos-p+1);
        logP = blm->BoffLnPid(ptr_input+max(tgpos+1-lm_order, 0), lm_buf_target[ni], tgpos-p);
        nb_ex_short++;
      }
      else
#endif
      {

        if (buf_target_wid[ni] == slist_len) {
            // request proba from back-off LM for words not in slist
            // the order of the back-off LM may be smaller than the one of the CSLM
            // -> this is resolved internally by the back-off class (the last words are used)
          int p;
          for (p = 0 ; (p < tgpos) && (NULL_WORD == ptr_input[p]) ; p++); // search for longest n-gram without NULL_WORD in the first place
          logP = blm->BoffLnPid(ptr_input+max(tgpos+1-lm_order, p), lm_buf_target[ni],min(lm_order, tgpos + 1 - p));
	  //printf("NN slist output=%e\n", optr[buf_target_wid[ni]]);
        }
        else {
            // get proba from CSLM
          if (renorm) {
	    // renormalize CSLM proba with back-off LM proba mass on the fly (this is very costly)
            REAL pmass=0.0;
            for (WordID w=0; w<slist_len; w++) pmass+=blm->BoffPid(ptr_input+max(tgpos+1-lm_order, 0), w, min(lm_order, tgpos + 1));
            //printf("      BLM pmass=%e\n", pmass);
	    logP = safelog(optr[buf_target_wid[ni]] / (1.0-optr[slist_len]) * pmass);
          }
          else {
            logP = safelog(optr[buf_target_wid[ni]]); // no error check on indices necessary here
          }
          //REAL logP2 = blm->BoffLnPid(ptr_input+max(tgpos+1-lm_order, 0), lm_buf_target[ni],min(lm_order, tgpos + 1));
          //printf("       CSLM: logP=%e,  ngra,=%e \n", logP, logP2);
          nb_ex_slist++;
        }
      }

      log_sum += logP;
      ptr_input += idim;  // next example in bunch at input
      optr += odim;  // next example in bunch at output
      if (fname) {
        fs << logP << endl;
      }
    }

    nb_ex += n;
  } while (data_available);

  printf(" - %d %d-gram requests, %d=%.2f%% short n-grams, %d=%.2f%% by back-off LM, %d=%5.2f%% predicted by CSLM\n",
         nb_ex, iaux+1,
         nb_ex_short, 100.0*nb_ex_short/nb_ex,
         nb_ex-nb_ex_short-nb_ex_slist, 100.0*(nb_ex-nb_ex_short-nb_ex_slist)/nb_ex,
         nb_ex_slist, 100.0*nb_ex_slist/nb_ex);

 
  REAL px = (nb_ex>0) ? exp(-log_sum / (REAL) nb_ex) : -1;
  printf("   cslm px=%.2f, ln_sum=%.2f, overall px=%.2f\n",
        (nb_ex_slist>0) ? exp(-log_sum_cslm / (REAL) nb_ex) : -1, log_sum, px);

  if (fname) fs.close();

  return px;
}


//**************************************************************************************
// information after finishing an epoch

void TrainerNgramSlist::InfoPost ()
{
  printf(" - epoch finished, %d examples seen in short-list (%5.2f%% of a total of %d) short input=%d (%5.2f%%) average CSLM perplexity: %.2f\n",
	nb_ex_slist, 100.0*nb_ex_slist/nb_ex, nb_ex, nb_ex_short, 100.0*nb_ex_short/nb_ex, err_train);
}

//**************************************************************************************
// request one n-gram probability, usually the call will be delayed
// and processed later 

void TrainerNgramSlist::BlockEval(WordID *wid, int o, REAL*p, REAL *aux_data)
{
    int cl=o-1, i;
    if (cl != iaux) {
#ifdef CSLM_DOES_SHORT_NGRAMS
      req[nreq].ctxt_len = iaux;  // use full filled-up n-gram
      req[nreq].ctxt = new WordID[iaux];
	// fill up incomplete n-gram with NULL-WORD (at the beginning !)
      for (i=0; i<iaux-cl; i++) req[nreq].ctxt[i]=NULL_WORD; 
      int newTgpos = tgpos+i;
      if (newTgpos > iaux) newTgpos=iaux;
      for (int j=0; i<iaux; i++, j++) 
      {
	if (i >= newTgpos){req[nreq].ctxt[i]=wid[j+1];} 
	else { req[nreq].ctxt[i]=wid[j]; }
        if (i == newTgpos) req[nreq].wpred = wid[j];
      }
      
      if (i == newTgpos) req[nreq].wpred = wid[cl];
      
      req[nreq].res_ptr = p;
      if ((NULL != aux_data) && (0 < auxdim)) {
        req[nreq].aux_len = auxdim;
        req[nreq].aux = new REAL[auxdim];
        for (int j=0; j<auxdim; j++) req[nreq].aux[j]=aux_data[j];
      }
      else {
        req[nreq].aux_len = 0;
        req[nreq].aux = NULL;
      }
      if (++nreq >= max_req) BlockFinish();
#else
      //ErrorN("BlockEval() dim %d differs from CSLM %d\n", cl, iaux);
      nb_ex_short++;
      *p = blm->BoffLnStd(wid, wid[cl], o);
#endif
      return;
    }

    req[nreq].ctxt_len = cl;
    req[nreq].ctxt = new WordID[cl];
    for(i=0;i<cl; i++) if (i >= tgpos) { req[nreq].ctxt[i]=wid[i+1];} else { req[nreq].ctxt[i]=wid[i]; }
    req[nreq].wpred = wid[tgpos];
    req[nreq].res_ptr = p;
    if ((NULL != aux_data) && (0 < auxdim)) {
      req[nreq].aux_len = auxdim;
      req[nreq].aux = new REAL[auxdim];
      for (int j=0; j<auxdim; j++) req[nreq].aux[j]=aux_data[j];
    }
    else {
      req[nreq].aux_len = 0;
      req[nreq].aux = NULL;
    }
    if (++nreq >= max_req) BlockFinish();
}

//**************************************************************************************
// 

int NgramReqComp(const void *v1, const void *v2)
{ NgramReq* n1=(NgramReq*) v1, *n2=(NgramReq*) v2;
     for (int i=0; i<n1->ctxt_len; i++) {
       if (n1->ctxt[i] < n2->ctxt[i]) return -1;
       if (n1->ctxt[i] > n2->ctxt[i]) return 1;
     }
     for (int i=0; i<n1->aux_len; i++) {
       if (n1->aux[i] < n2->aux[i]) return -1;
       if (n1->aux[i] > n2->aux[i]) return 1;
     }
     return 0; // both are equal
   }

//**************************************************************************************
// process all delayed n-gram requests

void TrainerNgramSlist::BlockFinish()
{
  if (nreq == 0) return;

  nb_ngram+=nreq;

#ifdef DEBUG
  for (int i=0; i<nreq; i++) {
    printf("buf %d: ", i); for (int c=0; c<req[i].ctxt_len; c++) printf(" %d", req[i].ctxt[c]);
    printf(" -> %d\n", req[i].wpred);
  }
#endif
  //sort(req.begin(),req.end());  // use operator < of Ngramreq
  qsort(req, nreq, sizeof(NgramReq), NgramReqComp);

#ifdef DEBUG
  for (int i=0; i<nreq; i++) {
    printf("buf %d: ", i); for (int c=0; c<req[i].ctxt_len; c++) printf(" %d", req[i].ctxt[c]);
    printf(" -> %d\n", req[i].wpred);
  }
#endif

  int n,i;

    // process first n-gram input of CSLM
  req[0].bs=0;
  for (i=0;i<req[0].ctxt_len; i++) {
    buf_input[i] = (REAL) wlist->MapIndex(req[0].ctxt[i]);      // map context words IDs
  }
  for (int j=0; j<req[0].aux_len; i++, j++) {
    buf_input[i] = req[0].aux[j];               // append auxiliary data
  }
	
    // add new n-gram inputs to CSLM if context changes
    // perform forward pass if bunch is full
    // ususally we need to do several forward bunches

  int req_beg=0;	// start of current CSLM block in large request array
  int bs=0;  		// current block index in forward bunch
  for (n=1; n<nreq; n++) {
    if (NgramReqComp(req+n-1, req+n) != 0) { 
      bs++;
      if (bs >= bsize) {
        ForwAndCollect(req_beg,n-1,bs,false);
        bs=0; req_beg=n;
      }

      req[n].bs=bs;
      for (i=0;i<req[n].ctxt_len; i++) {
        buf_input[bs*idim+i] = (REAL) wlist->MapIndex(req[n].ctxt[i]);  // map context words IDs
      }
      for (int j=0; j<req[n].aux_len; i++, j++) {
        buf_input[bs*idim+i] = req[n].aux[j];         // append auxiliary data
      }
    }
    else
      req[n].bs=bs;
  }
  ForwAndCollect(req_beg,nreq-1,bs+1,false);
  FreeReq();
}


//**************************************************************************************
// collect all delayed probability requests

void TrainerNgramSlist::ForwAndCollect(int req_beg, int req_end, int bs, bool renorm)
{
  if (bs<=0) return;
  nb_forw++;
#ifdef CUDA
  Gpu::SetConfig(mach->GetGpuConfig());
  mach->SetDataIn(gpu_input);
  Gpu::MemcpyAsync(gpu_input, buf_input , bs*idim*sizeof(REAL), cudaMemcpyHostToDevice);
#else
  mach->SetDataIn(buf_input);
#endif
  mach->Forw(bs,false); //TODO

    // collect the outputs and store them at the provided addresses
#ifndef CSLM_DOES_SHORT_NGRAMS
  WordID mapped_bos = wlist->MapIndex(wlist->GetIndex(WordList::WordSentStart));
#endif
  int lm_order=blm->GetOrder();

#ifdef BLAS_CUDA
  Gpu::MemcpyAsync(host_output, mach->GetDataOut(), bs*odim*sizeof(REAL), cudaMemcpyDeviceToHost);
  Gpu::StreamSynchronize();
#endif

  for (int n=req_beg; n<=req_end; n++) {
    REAL logP=0;
    WordID tgt = req[n].wpred;
    if (tgt==NULL_WORD) Error("TrainerNgramSlist::ForwAndCollect(): internal error: NULL_WORD in target\n");
    WordID mapped_tgt = wlist->MapIndex(tgt);
    int b=req[n].bs;
#ifdef BLAS_CUDA
    REAL *optr=host_output + b*odim;
#else
    REAL *optr=mach->GetDataOut() + b*odim;
#endif
    REAL *ptr_input = buf_input + b*idim;

        // if the current n-gram has a BOS elsewhere than in the first place -> find the shortest n-gram order and request it
        // DataNext() will take care to propose the next longer n-gram
        // TODO: check what happens if there more BOS and EOS are in the rest of the n-gram 
#ifndef CSLM_DOES_SHORT_NGRAMS
      if ((idim>1) && ((WordID) ptr_input[1] == mapped_bos)) {
        int p;
        Error("TrainerNgramSlist::ForwAndCollect(): <s> in the middle of an n-gram");
        for (p=tgpos-1; p>=1 && (ptr_input[p]!=mapped_bos); p--);  // Walid:is this really a loop do to ";"?
          // Walid: We need to check if is it correct to send tgt or mapped_tgt
          logP = blm->BoffLnPid(ptr_input+max(tgpos+1-lm_order, 0), mapped_tgt, tgpos-p+1);
      }
      else
#endif
      {
        if (!wlist->InShortList(mapped_tgt)) {
            // request proba from back-off LM for words not in slist
            // the order of the back-off LM may be smaller than the one of the CSLM
            // -> this is resolved internally by the back-off class (the last words are used)
          logP = blm->BoffLnPid(ptr_input+max(tgpos+1-lm_order, 0), mapped_tgt, min(lm_order, tgpos + 1)); // TODO target mapped forth an back
        }
        else {
            // get proba from CSLM
          if (renorm) {
	    // renormalize CSLM proba with back-off LM proba mass on the fly (this is very costly)
            REAL pmass=0.0;
            for (WordID w=0; w<slist_len; w++) pmass+=blm->BoffPid(ptr_input+max(tgpos+1-lm_order, 0), w, min(lm_order, tgpos + 1));
	    logP = safelog(optr[mapped_tgt] / (1.0-optr[slist_len]) * pmass);
          }
          else {
            logP = safelog(optr[mapped_tgt]); // no error check on indices necessary here
          }
          nb_ex_slist++;
        }
      }

        // store LM proba
      *(req[n].res_ptr) = logP;
    } // for (ni=...)

}

//**************************************************************************************
// 
void TrainerNgramSlist::BlockSetMax(int p_max) {
  if (req) {
    FreeReq();
    delete [] req;
  }
  max_req=p_max;
  req = new NgramReq[max_req];
  nreq=0;
}


//**************************************************************************************
// information after finishing an epoch

void TrainerNgramSlist::BlockStats() {
   printf(" - %d %d-gram requests, %d=%.2f%% short n-grams, %d=%5.2f%% predicted by CSLM, %d forward passes (avrg of %d probas)\n",
	nb_ngram, iaux+1, nb_ex_short, 100.0*nb_ex_short/nb_ngram, nb_ex_slist, 100.0*nb_ex_slist/nb_ngram, nb_forw, nb_ngram/nb_forw);
}
