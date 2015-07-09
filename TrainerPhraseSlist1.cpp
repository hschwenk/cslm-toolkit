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
#include "MachSplit1.h"
#include "TrainerPhraseSlist1.h"

#include "NBest.h" 
#include "sort.cpp" 

void TrainerPhraseSlist1::DoConstructorWork()
{
  char	msg[1024];

  idim=mach->GetIdim(); odim=mach->GetOdim(); bsize=mach->GetBsize();

#ifdef BLAS_CUDA
  Gpu::SetConfig(mach->GetGpuConfig());
  gpu_input = Gpu::Alloc(idim*bsize, "inputs in Trainer");
  gpu_target = Gpu::Alloc(odim*bsize, "targets in Trainer");
  host_output = new REAL[odim*bsize];
#endif
  buf_target_wid = new WordID[odim*bsize];
  buf_target_ext = new WordID[odim*bsize];

    // set up vector to outputs of the target phrases
  if (mach->GetMType() != file_header_mtype_mseq)
    Error("CSTM: sequential machine needed\n");
  MachSeq *mseq=(MachSeq*) mach;
  if (mseq->MachGetNb()<2)
    Error("CSTM: the number of machines is suspeciously small");

    // check input layer
  if (mseq->MachGet(0)->GetMType() != file_header_mtype_mpar)
    Error("TrainerPhraseSlist1::DoConstructorWork: CSTM: the input layer has the wrong architecture\n");
  MachPar *mpar = (MachPar*) mseq->MachGet(0);
  if (mpar->MachGet(0)->GetMType() != file_header_mtype_tab)
    Error("TrainerPhraseSlist1::DoConstructorWork: CSTM: the input layer has the wrong architecture\n");
  MachTab *mtab = (MachTab*) mpar->MachGet(0);
  max_inp_idx = mtab->GetMaxInpVal();

    // check output layer
  if (mseq->MachGet(mseq->MachGetNb()-1)->GetMType() != file_header_mtype_msplit1)
    Error("CSTM: the output layer has the wrong architecture\n");
  MachSplit1 *msp = (MachSplit1*) mseq->MachGet(mseq->MachGetNb()-1);
  tg_nbphr=msp->MachGetNb();
  if (data_train && (data_train->GetOdim() != tg_nbphr)) {
    sprintf(msg,"CSTM: output dimension of the training data should be %d, found %d\n", tg_nbphr, data_train->GetOdim());
    Error(msg);
  }

  phrase_mach.clear();
  for (int m=0; m<tg_nbphr; m++) {
    phrase_mach.push_back(msp->MachGet(m));
    if (m>0 && phrase_mach[m-1]->GetOdim() != phrase_mach[m]->GetOdim())
      Error("CSTM: the output layer dimension must be identical for all phrases\n");
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
    Error("the size of the source word list exceeds the number of input words the machine was trained for");

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
    Error("the target word list don't contain word count");
  if (tg_wlist->GetSize() <= tg_slist_len)
    Error("TrainerPhraseSlist1: the output layer is larger than the target word list");

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
  
  debug0(" + done init TrainerPhraseSlist1\n");
}

//
// constructor for training
//

TrainerPhraseSlist1::TrainerPhraseSlist1 (Mach *pmach, Lrate *lrate, ErrFct *perrfct,
	const char *train_fname, const char *dev_fname, const char *pt_fname, int p_nscores,
	REAL p_wd, int p_maxep, int p_ep)
 : Trainer(pmach,lrate,perrfct,NULL,NULL,p_wd,p_maxep,p_ep),
   tg_nbphr(0), tg_slist_len(0), 
   sr_wlist(NULL), tg_wlist(NULL),
   nb_ex_slist(0), nb_ex_short(0),
   nb_forw(0)
{
  debug2("*** Constructor TrainerPhraseSlist1 for training idim=%d, odim=%d ***\n",idim,odim);
  cout << "Setting up CSTM training with short list" << endl;
  char msg[1024];

  if (train_fname) {
    data_train = new Data(train_fname);
    if (idim != data_train->GetIdim()) {
      sprintf(msg,"TrainerPhraseSlist1: input dimension of the training data (%d) does not match the one of the machine (%d)\n", data_train->GetIdim(), idim);
      Error(msg);
    }
    if (data_train->GetOdim()<1 || data_train->GetOdim()>10) {
      sprintf(msg,"TrainerPhraseSlist1: output dimension of the training data should be 1..10, found %d\n", data_train->GetOdim());
      Error(msg);
    }
    auxdim = data_train->GetAuxdim();
  }
  else 
    data_train=NULL;

  if (dev_fname) {
    data_dev = new Data(dev_fname);
    data_dev_alloc=true;
    if (idim != data_dev->GetIdim()) {
      sprintf(msg,"TrainerPhraseSlist1: input dimension of the validation data (%d) does not match the one of the machine (%d)\n", data_dev->GetIdim(), idim);
      Error(msg);
    }
    if (data_dev->GetOdim()<1 || data_dev->GetOdim()>10) {
      sprintf(msg,"TrainerPhraseSlist1: output dimension of the validation data should be 1..10, found %d\n", data_dev->GetOdim());
      Error(msg);
    }
    int auxdim_dev = data_dev->GetAuxdim();
    if (0 >= auxdim)
      auxdim = auxdim_dev;
    else if (auxdim != auxdim_dev)
      ErrorN("TrainerPhraseSlist1: auxiliary data dimension of the validation data should be %d, found %d", auxdim, auxdim_dev);
  }
  else {
    data_dev=NULL;
    data_dev_alloc=false;
  }
  iaux = (idim - auxdim);

  DoConstructorWork();

  if (data_dev) {
    if (pt_fname) {
      cout << " - loading external phrase table from " << pt_fname << endl;
      ptable.Read(pt_fname,5,"1:2");
    }
    else
      cout << " - no external phrase table provided" << endl;
  }
}

//
// constructor for testing
//

TrainerPhraseSlist1::TrainerPhraseSlist1 (Mach *pmach, ErrFct *perrfct,
	Data *data, char *pt_fname, int p_nscores)
 : Trainer(pmach,NULL,perrfct,NULL,NULL),
   tg_nbphr(0), tg_slist_len(0), 
   sr_wlist(NULL), tg_wlist(NULL),
   nb_ex_slist(0), nb_ex_short(0),
   nb_forw(0)
{
  debug0("*** Constructor TrainerPhraseSlist1 for testing ***\n");
  cout << "Setting up testing with short list" << endl;
  char	msg[1024];

  data_train=NULL;
  data_dev=data;
  data_dev_alloc=false; // do not free it by this class !

  if (idim != data_dev->GetIdim()) {
    sprintf(msg,"TrainerPhraseSlist1: input dimension of the test data (%d) does not match the one of the machine (%d)\n", data_dev->GetIdim(), idim);
    Error(msg);
  }
  auxdim = data_dev->GetAuxdim();
  iaux = (idim - auxdim);

  DoConstructorWork();

  cout << " - loading external phrase table from " << pt_fname << endl;
#ifdef BACKWRAD_TM
  ptable.Read(pt_fname,5,"1:0"); // backward TM prob
#else
  ptable.Read(pt_fname,5,"1:2"); // forward TM prob
#endif
}

//
// constructor for nbest rescoring
//

TrainerPhraseSlist1::TrainerPhraseSlist1 (Mach *pmach,
    WordList *p_sr_wlist, WordList *p_tg_wlist,
	char *pt_fname, int nscores, char *scores_specif)
 : Trainer(pmach,NULL,NULL,NULL,NULL), // TODO; should I call:  TrainerNgram(pmach,NULL,NULL,NULL),
   tg_nbphr(0), tg_slist_len(0), 
   sr_wlist(p_sr_wlist), tg_wlist(p_tg_wlist),
   nb_ex_short(0), nb_forw(0)
{
  debug0("*** Constructor TrainerPhraseSlist1 for block operations ***\n");
  cout << "Setting up CSTM with short list" << endl;
  // TODO: init with TrainerNgram before
  DoConstructorWork();

  cout << " - loading external phrase table from " << pt_fname << endl;
  ptable.Read(pt_fname, nscores, scores_specif);
}

//**************************************************************************************

TrainerPhraseSlist1::~TrainerPhraseSlist1 ()
{ 
  debug0("*** Destructor TrainerPhraseSlist1 ***\n");

  if (buf_target_wid) delete [] buf_target_wid;
  if (buf_target_ext) delete [] buf_target_ext;
    // buf_input and buf_target will be deleted by ~Trainer()

  phrase_mach.clear();

#ifdef DEBUG2
  vector<char*>::const_iterator iter, end = words.end();
  for (iter=words.begin(); iter!=end; iter++) delete *iter;
  words.clear();
#endif
}


//**************************************************************************************

REAL TrainerPhraseSlist1::Train()
{
  if (!data_train) return -1;
#ifdef DEBUG
  printf("*****************\n");
  printf("TrainerPhraseSlist1::Train():\n");
  printf(" -    idim=%d, odim=%d, tg_nbphr=%d\n", idim, odim, tg_nbphr);
  printf(" -  data_in: %p \n", (void*) buf_input);
  printf(" -   target: %p \n", (void*) buf_target);
  printf(" -  tgt WID: %p \n", (void*) buf_target_wid);
  printf(" - grad_out: %p \n", (void*) errfct->GetGrad());
#endif

  Timer ttrain;		// total training time
  Timer tload;
  Timer ttransfer;      // total transfer time of data to GPU
  Timer tforw;          // total forw time
  Timer tgrad;          // total gradient time
  Timer tbackw;         // total backw time
  ttrain.start();

  data_train->Rewind();

  REAL log_sum=0;
  int i;
  nb_ex=nb_ex_slist=nb_ex_short_inp=nb_ex_short=0;
  nb_tg_words=nb_tg_words_slist=0;


#ifdef BLAS_CUDA
  Gpu::SetConfig(mach->GetGpuConfig());
  mach->SetDataIn(gpu_input);		// we copy from buf_input to gpu_input
  errfct->SetTarget(gpu_target);	// we copy from buf_target to gpu_target
  debug1(" - gpu_input %p\n", gpu_input);
  debug1(" - gpu_target %p\n", gpu_target);
#else
  mach->SetDataIn(buf_input);
  errfct->SetTarget(buf_target);
  debug1(" - buf_input %p\n", buf_input);
  debug1(" - buf_target %p\n", buf_target);
#endif
  errfct->SetOutput(mach->GetDataOut());
  mach->SetGradOut(errfct->GetGrad());
  bool data_available;
  do {
    tload.start();
      // get a bunch of data and map all the words
    int n=0, nbtgsl=0;
    data_available = true;
    while (n < mach->GetBsize() && data_available) {
      data_available = data_train->Next();
      if (!data_available) break;
      debug0("TRAIN DATA: input: ");
      bool at_least_one_short=false;
      for (i=0; i<iaux; i++) { // copy word indexes
        WordID inp=(WordID) data_train->input[i];
        debug2(" %s[%d]", sr_wlist->GetWordInfo(inp).word,inp);
#if TODO // should we map input data ?
        buf_input[n*idim + i] = (REAL) sr_wlist->MapIndex(inp, "TrainerPhraseSlist1::Train(): input");       // map context words IDs
        if (inp == NULL_WORD)
          at_least_one_short=true;
#else
        buf_input[n*idim + i] = inp;
        if (inp == NULL_WORD)
          at_least_one_short=true;
        else if (inp<0 || inp>=(int)sr_wlist->GetSize())
          ErrorN("TrainerPhraseSlist1::Train(): input out of bounds (%d), must be in [0,%d[", inp, (int) sr_wlist->GetSize());
#endif
      }
      for (; i < idim ; i++) // copy auxiliary data
        buf_input[n * idim + i] = data_train->input[i];
      if (at_least_one_short) nb_ex_short_inp++;

      debug0(" - > mapped: ");
      
      bool all_in_slist=true;  // ALL to be predicted words are in short list
      at_least_one_short=false;
      nbtgsl=0;
      for (i=0; i<tg_nbphr; i++) {
        WordID outp=(WordID) data_train->target[i];
        int idx=i+n*tg_nbphr;
        buf_target_wid[idx] = tg_wlist->MapIndex(outp, "TrainerPhraseSlist1::Train(): output");     // TODO: not really needed during training, just the current value
        if (outp==NULL_WORD) {
          buf_target[idx] = (REAL) NULL_WORD;
          at_least_one_short=true;
          debug1(" -[%d->NULL]",(int) buf_target[idx]);
        }
        else {
          nb_tg_words++;
          if (tg_wlist->InShortList(buf_target_wid[idx])) {
            buf_target[idx] = (REAL) buf_target_wid[idx];
            debug3(" %s[%d->%d]", tg_wlist->GetWordInfo(outp).word,(int) buf_target_wid[idx], outp);
            nbtgsl++;
          }
          else {
	    buf_target[idx] = (REAL) tg_slist_len;	// words that are not in slist are ALL done by the last output neuron
            debug3(" %s[%d->%d]*", tg_wlist->GetWordInfo(outp).word,(int) buf_target_wid[idx], outp);
            all_in_slist=false;
          }
        }
      }
      if (all_in_slist) {
        nb_ex_slist++;
        nb_tg_words_slist += nbtgsl;
      }
      if (at_least_one_short) nb_ex_short++;
      debug1("     all_slist=%d\n",all_in_slist);

      n++;
    }  // loop to get a bunch of examples
    debug4("train bunch of %d words, totl=%d, totl slist=%d [%.2f%%]\n", n, nb_ex+n, nb_ex_slist, 100.0*nb_ex_slist/(nb_ex+n));
    tload.stop();

#ifdef DEBUG2
printf("network data:\n");
REAL *iptr=buf_input;
REAL *tptr=buf_target;
for (int nn=0;nn<n;nn++) {
   for (i=0;i<idim;i++) printf(" %f", *iptr++); printf(" -> ");
   for (i=0;i<tg_nbphr;i++) printf(" %f", *tptr++); printf("\n");
}
#endif

    if (n>0) {
#ifdef BLAS_CUDA
      ttransfer.start();
      Gpu::MemcpyAsync(gpu_input, buf_input , n*idim*sizeof(REAL), cudaMemcpyHostToDevice);
      Gpu::MemcpyAsync(gpu_target, buf_target , n*odim*sizeof(REAL), cudaMemcpyHostToDevice);
      Gpu::StreamSynchronize();
      ttransfer.stop();
#endif
      tforw.start();
      mach->Forw(n,true);
      tforw.stop();

      tgrad.start();
      log_sum += errfct->CalcGrad(n);
      tgrad.stop();

      debug1("  log_sum=%e\n",log_sum);
#ifdef DEBUG2
      int t=(int) data_train->target[0];
#ifdef BLAS_CUDA
      Gpu::SetConfig(mach->GetGpuConfig());
      REAL * tmp = Gpu::Alloc(5, "tmp buffer for DEBUG2");
      cublasGetVector(odim,CUDA_SIZE,mach->GetDataOut(),1,tmp,1);
      printf("OUTPUT:");
      for (int i=t-2;i<=t+2; i++) printf(" %f",tmp[i]); printf("\n");
      cublasGetVector(3, CUDA_SIZE, data_train->target, 1, tmp, 1);
      printf("TARGET:");
      for (int i=0;i<1; i++) printf(" %f", tmp[i]); printf("\n");
      //TODO check if we need odim or idim!
      cublasGetVector(odim*bsize, CUDA_SIZE, errfct->GetGrad(), 1, tmp, 1);
      printf("  GRAD:");
      for (int i=t-2;i<=t+2; i++) printf(" %f",tmp[i]); printf("\n");
      cublasFree(tmp);
#else
printf("OUTPUT:") ; for (int i=t-2;i<=t+2; i++) printf(" %f",mach->GetDataOut()[i]); printf("\n");
printf("TARGET:") ; for (int i=0;i<1; i++) printf(" %f",data_train->target[i]); printf("\n");
printf("  GRAD:") ; for (int i=t-2;i<=t+2; i++) printf(" %f",errfct->GetGrad()[i]); printf("\n");
#endif //BLAS_CUDA
#endif //DEBUG2
      lrate->UpdateLrateOnForw(mach->GetNbForw());
      tbackw.start();
      mach->Backw(lrate->GetLrate(), wdecay, n);
      tbackw.stop();
    }

    nb_ex += n;
  } while (data_available);
#ifdef BLAS_CUDA
  Gpu::StreamSynchronize();
#endif

  ttrain.stop();
  ttrain.disp(" - training time: ");
  tload.disp(" including load: ");
#ifdef BLAS_CUDA
  ttransfer.disp(" transfer: ");
#endif
  tforw.disp(" forw: ");
  tgrad.disp(" grad: ");
  tbackw.disp(" backw: ");
  printf("\n");
  
  printf(" = log_sum=%.2f, nb_tg_words=%d, nb_ex_slist=%d, nb_tg_words_slist=%d\n", log_sum, nb_tg_words, nb_ex_slist, nb_tg_words_slist);
  if (nb_tg_words>0) return exp(-log_sum / (REAL) nb_tg_words);  // when normalizing consider that all examples lead to a forward pass 

  return -1;
}

//**************************************************************************************
// 

void TrainerPhraseSlist1::GetMostLikelyTranslations (ofstream &fspt, REAL *optr, int ni)
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
void TrainerPhraseSlist1::GetMostLikelyTranslations (ofstream &fspt, REAL *optr, int ni)
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

REAL TrainerPhraseSlist1::TestDev(char *fname)
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

  char *ptfname = (char*) "alltrans.txt";
  ofstream fspt;
  cout << " - dumping new phrase table to file '" << ptfname << "'" << endl;
  fspt.open(ptfname,ios::out);
  CHECK_FILE(fspt,ptfname);

  nb_ex=nb_ex_slist=nb_ex_short=0;
  nb_tg_words=nb_tg_words_slist=0;
  int nb_probs=0;	// this counts the number of cumulated log probs.
			// This increments by only one for external phrase tables, independently of the target phrase length
  REAL logP, log_sum=0;
  REAL log_sum_cstm=0;	// only CSLM, i.e. considering phrases done by CSTM

  uint idx;

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
        debug1(" %d", inp);
#if TODO // should we map input data ?
        buf_input[idx] = (REAL) sr_wlist->MapIndex(inp, "TrainerPhraseSlist1::TestDev(): input");       // map context words IDs
        if (inp == NULL_WORD)
          at_least_one_short=true;
#else
        buf_input[idx] = inp;
        if (inp == NULL_WORD)
          at_least_one_short=true;
        else if (inp<0 || inp>=(int)sr_wlist->GetSize())
          ErrorN("TrainerPhraseSlist1::TestDev(): input out of bounds (%d), must be in [0,%d[", inp, (int) sr_wlist->GetSize());
#endif
      }
      for (; i < idim ; i++) // copy auxiliary data
        buf_input[n * idim + i] = data_dev->input[i];
      if (at_least_one_short) nb_ex_short_inp++;

      debug0(" - > mapped: ");
      
      bool all_in_slist=true;  // ALL to be predicted words are in short list
      int nb_words_not_null=0;
      at_least_one_short=false;
      for (i=0; i<tg_nbphr; i++) {
        WordID outp=(WordID) data_dev->target[i];
        idx=n*tg_nbphr + i;
        buf_target_wid[idx] = tg_wlist->MapIndex(outp, "TrainerPhraseSlist1::TestDev(): output");
        buf_target_ext[idx] = buf_target_wid[idx];		// keep target word ID for Moses phrase-table
        if (outp==NULL_WORD) {
          buf_target[idx] = (REAL) NULL_WORD;
          at_least_one_short=true;			// TODO: optimize: we should be able to stop the loop on "i"
          debug1(" %d[NULL]",(int) buf_target_wid[idx]);
        }
        else {
          nb_tg_words++;
          nb_words_not_null++;
          if (tg_wlist->InShortList(buf_target_wid[idx])) {
            buf_target[idx] = (REAL) buf_target_wid[idx];
            debug3(" %s[%d->%d]", tg_wlist->GetWordInfo(outp).word, (int) buf_target_wid[idx], outp);
	    //nbtgsl++;
          }
          else {
	      // TODO: we actually don't need a forward pass for words in the short lists or short n-grams
	      //       this could be used to save some time (5-10%)
            buf_target_wid[idx] = tg_slist_len;
	    buf_target[idx] = (REAL) tg_slist_len;	// words that are not in slist are ALL done by the last output neuron
            debug3(" %s[%d->%d]*", tg_wlist->GetWordInfo(outp).word,(int) buf_target_wid[idx], outp);
            all_in_slist=false;
          }
        }
      }
      done_by_cstm.push_back(all_in_slist);
      if (all_in_slist) {
        nb_ex_slist++;
        nb_tg_words_slist += nb_words_not_null;
        //nb_tg_words_slist += nbtgsl;
      }
      if (!at_least_one_short) nb_ex_short++;
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
#ifdef BLAS_CUDA
      Gpu::MemcpyAsync(gpu_input, buf_input , n*idim*sizeof(REAL), cudaMemcpyHostToDevice);
      Gpu::MemcpyAsync(gpu_target, buf_target , n*odim*sizeof(REAL), cudaMemcpyHostToDevice);
#endif
      mach->Forw(n,false); 
      log_sum_cstm += errfct->CalcValue(n);
    }

      // get probas from CSLM or back-off LM
#ifdef BLAS_CUDA
    cudaMemcpy(host_output, mach->GetDataOut(), n*odim*sizeof(REAL), cudaMemcpyDeviceToHost);
    REAL *optr=host_output;
    Error("TrainerPhraseSlist1::TestDev TODO CUDA");
#else
    REAL *optr=mach->GetDataOut();	// n times (tg_nbphr*tg_slen) = odim values
#endif

    debug1("Collect n=%d\n", n);
    if (n!=(int) done_by_cstm.size())
      Error("TrainerPhraseSlist1::TestDev(): internal error, number of phrases done by CSTM does not match");

    REAL *ptr_input = buf_input;	// n times idim values
    for (int ni=0; ni<n; ni++) {
      int nb_tg=0; // for normalization
      if (done_by_cstm[ni]) {
          // get proba from CSTM (removed renorm)
          
#define DUMP_PHRASE_TABLE
#ifdef DUMP_PHRASE_TABLE
          // create output phrase table
        for (i=0;i<iaux;i++) {
          if (buf_input[ni*idim+i] == NULL_WORD) break;
          fspt << sr_wlist->GetWordInfo(buf_input[ni*idim+i]).word << " ";
        }
        fspt << "||| ";
        for (i=0;i<tg_nbphr;i++) {
          if (buf_target_wid[i+ni*tg_nbphr] == NULL_WORD) break;
          fspt << tg_wlist->GetWordInfoMapped(buf_target_wid[ni*tg_nbphr+i]).word << " ";
        }
        fspt << "||| ";
#endif

        logP=0;
        REAL *optr2=optr;
        for (i=0; i<tg_nbphr; i++) {
          if (buf_target_wid[i+ni*tg_nbphr] == NULL_WORD) break;
          logP += safelog(optr2[buf_target_wid[i+ni*tg_nbphr]]); // no error check on indices necessary here
          nb_tg++;
#ifdef DUMP_PHRASE_TABLE2
	  fspt << optr2[buf_target_wid[i+ni*tg_nbphr]] << " ";  
#endif
          optr2+=dim_per_phrase;
        }
#ifdef DUMP_PHRASE_TABLE
        fspt << logP/nb_tg << endl;
#endif

#ifdef DUMP_PHRASE_TABLE_NBEST
        GetMostLikelyTranslations(fspt,optr,ni);
#endif

        nb_probs+=i;
        debug1(" CSLM: logP=%e\n", logP);
      }
      else {
          // request proba from Moses phrase-table
#if 1
        debug0("create textual phrase pair for external phrase table (word + index)\n");
        src_phrase.clear();
        debug0("  source:");
        for (i=0; i<iaux && ptr_input[i]!=NULL_WORD; i++) {
          src_phrase.push_back(sr_wlist->GetWordInfo((uint) ptr_input[i]).word);	// TODO: char* to string
          debug2(" %s[%d]", src_phrase.back().c_str(), (uint) ptr_input[i]);
        }
        tgt_phrase.clear();
        debug0("  target:");
        for (i=0; i<tg_nbphr && buf_target_ext[i+ni*tg_nbphr]!=NULL_WORD; i++) {
          tgt_phrase.push_back(tg_wlist->GetWordInfoMapped(buf_target_ext[i+ni*tg_nbphr]).word);	// TODO: char* to string
          debug2(" %s[%d]", tgt_phrase.back().c_str(), buf_target_ext[i+ni*tg_nbphr]);
        }
#ifdef BACKWRAD_TM
        logP = safelog(ptable.GetProb(tgt_phrase, src_phrase));
#else
        logP = safelog(ptable.GetProb(src_phrase, tgt_phrase));
#endif
        nb_probs++;
        debug1("  => logP=%e\n",logP);
#else
        logP=1;
#endif
      }

      log_sum += logP;
      ptr_input += idim;  // next example in bunch at input
      optr += odim;  // next example in bunch at output
      if (fname) {
        fs << ((nb_tg>0) ? logP/nb_tg : -1) << endl;
      }
    }

    nb_ex += n;
    debug2("%d: %f\n",nb_ex,exp(-log_sum/nb_ex));
  } while (data_available);

  printf(" %d target words in %d phrases (%d=%.2f%% uncomplete), CSTM: %d target words in %d phrases (%.2f%%)\n",
         nb_tg_words, nb_ex, 
         nb_ex_short, 100.0*nb_ex_short/nb_ex,
         nb_tg_words_slist, nb_ex_slist, 100.0*nb_ex_slist/nb_ex);

 
  REAL px = (nb_probs>0) ? exp(-log_sum / (REAL) nb_probs) : -1;
  printf("   cstm px=%.2f, ln_sum=%.2f, overall px=%.2f (%d values)\n",
        (nb_tg_words_slist>0) ? exp(-log_sum_cstm / (REAL) nb_tg_words_slist) : -1, log_sum_cstm, px, nb_probs);

  if (fname) fs.close();
  fspt.close();

  return px;
}


//**************************************************************************************
// information after finishing an epoch

void TrainerPhraseSlist1::InfoPost ()
{
  printf(" - epoch finished, %d target words in %d phrases (%.2f/%.2f%% short source/target)\n",
	nb_tg_words, nb_ex,
	100.0*nb_ex_short_inp/nb_ex_slist, 100.0*nb_ex_short/nb_ex_slist);
  printf("   CSTM: %d target words in %d phrases (%.2f%%), avrg px=%.2f\n",
	nb_tg_words_slist, nb_ex_slist, 100.0*nb_ex_slist/nb_ex,
	err_train);
}

//**************************************************************************************
// request one n-gram probability, usually the called will be delayed
// and processes later 


//**************************************************************************************
// collect all delayed probability requests


void TrainerPhraseSlist1::ForwAndCollect(vector< vector<string> > &src_phrases, AlignReq *areq, int req_beg, int req_end, int bs, int tm_pos)
{
  if (bs<=0) return;
  debug3("TrainerPhraseSlist1::ForwAndCollect(): collecting outputs %d .. %d from bunch of size %d\n", req_beg, req_end, bs);
  debug3("\ttarget machines %d x dim %d = total %d\n", tg_nbphr, dim_per_phrase, odim);

  if (bs != (int) src_phrases.size())
    ErrorN("TrainerPhraseSlist1::ForwAndCollect(): the number of source phrases (%d) does not match block length (%d)", (int) src_phrases.size(), bs);

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
  Gpu::MemcpyAsync(host_output, mach->GetDataOut(), bs*odim*sizeof(REAL), cudaMemcpyDeviceToHost);
  Gpu::StreamSynchronize();
#endif

    // stats
  int cnt_ex_slist=0, cnt_tg_words=0, cnt_tg_words_slist=0;

  for (int n=req_beg; n<=req_end; n++) {
    REAL logP=0;
    int b=areq[n].bs;

    if ((int) areq[n].tgph.size() > tg_nbphr)
      ErrorN("TrainerPhraseSlist1::ForwAndCollect(): target phrase too long (%d) for machine (%d)", (int) areq[n].tgph.size(), tg_nbphr);

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
      if (outp==NULL_WORD) break;
      cnt_tg_words++;
      buf_target_wid[tw] = tg_wlist->MapIndex(outp, "TrainerPhraseSlist1::ForwAndCollect() output");
      debug1("->%d",buf_target_wid[tw]);
      all_in_slist=tg_wlist->InShortList(buf_target_wid[tw]);
    }
      // fill up
    for (; tw<tg_nbphr; tw++) {
      debug0(" fill");
      buf_target_wid[tw]=NULL_WORD;
    }
    debug1("    slist=%d\n",all_in_slist);

#ifdef BLAS_CUDA
    REAL *optr=host_output + b*odim;
#else
    REAL *optr=mach->GetDataOut() + b*odim;
#endif

    if (!all_in_slist) {
        // get proba from external phrase table
      logP=ptable.GetProb(src_phrases[areq[n].bs], areq[n].tgph);
      debug1(" ptable: logP=%f\n", logP);
    }
    else {
        // get proba from CSLM
      debug0(" -  in slist CSLM:");
      logP=0; int cnt=0;
      for (int tw=0; tw<tg_nbphr; tw++) {
        if (buf_target_wid[tw] == NULL_WORD) break;
        debug1(" %e", optr[buf_target_wid[tw]]);
        logP += safelog(optr[buf_target_wid[tw]]);
        optr+=dim_per_phrase;
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


void TrainerPhraseSlist1::BlockStats() {
   //printf(" - %d phrase probability requests, %d=%5.2f short phrase %d forward passes (avrg of %d probas), %d=%5.2f%% predicted by CSTM\n",
	//nb_ngram, nb_ex_short, 100.0*nb_ex_short/nb_ngram, nb_forw, nb_ngram/nb_forw, nb_ex_slist, 100.0*nb_ex_slist/nb_ngram);
   printf(" - CSTM: %d forward passes, %d=%5.2f%% phrases were predicted by CSTM\n",
	nb_forw, nb_ex_slist, 100.0 * nb_ex_slist/nb_ex);
}
