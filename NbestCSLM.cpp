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


#include <iostream>
#include "Hypo.h"
#include "NbestCSLM.h"

#undef DUMMY_MACHINE
#ifdef DUMMY_MACHINE
 #include "Mach.h"
 #include "MachTab.h"
 #include "MachTanh.h"
 #include "MachSoftmax.h"
 #include "MachSeq.h"
 #include "MachPar.h"
#endif

NbestCSLM::~NbestCSLM() {
  if (mach) delete mach;
  if (trainer) delete trainer;
  for (vector<HypSentProba*>::iterator i = delayed_hyps.begin(); i != delayed_hyps.end(); i++) {
    delete (*i);
  }
  delayed_hyps.clear();
}


bool NbestCSLM::Read(char *fname, char *wl_fname, char *lm_fname, int tgpos, int aux_dim)
{
#ifdef DUMMY_MACHINE
  cout << " - rescoring with dummy CSLM " << endl;
  //int ctxt_size=4, idim=1000, pdim=256, hdim=128, slsize=1024, bsize=128;
  int ctxt_size=3, idim=1000, pdim=256, hdim=128, slsize=1, bsize=128;
  MachSeq *mseq = new MachSeq();
  MachPar *mp = new MachPar();
  MachTab *mt = new MachTab(idim,pdim,bsize);
  mt->TableRandom(0.1); mp->MachAdd(mt);
  REAL *tab_adr=mt->GetTabAdr();
  for (int i=1; i<ctxt_size; i++) {
    mt = new MachTab(tab_adr,idim,pdim,bsize);
    mp->MachAdd(mt);
  }
  mseq->MachAdd(mp);
  MachTanh *mh = new MachTanh(ctxt_size*pdim,hdim,bsize);
  mh->WeightsRandom(0.1); mh->BiasRandom(0.1);
  mseq->MachAdd(mh);
  MachSoftmax *mo = new MachSoftmax(hdim,slsize,bsize);
  mo->WeightsRandom(0.1); mo->BiasRandom(0.1);
  mseq->MachAdd(mo);
  mach=mseq;
#else
  ifstream ifs;
  ifs.open(fname,ios::binary);
  CHECK_FILE(ifs,fname);
  mach = Mach::Read(ifs);
  ifs.close();
#endif

  mach->Info();
  //lm_order = ((0 <= tgpos) ? tgpos : (mach->GetIdim() - aux_dim)) +1;
  lm_order = (mach->GetIdim() - aux_dim) +1; 
  // read word list
  cout << " - reading word list from file " << wl_fname;
  wlist.SetSortBehavior(this->stable_sort);
  WordList::WordIndex voc_size = wlist.Read(wl_fname);
  cout << endl;
#ifdef LM_KEN
  cout << " - using KENLM vocabulary with " << voc_size << " words\n";
#endif
#ifdef LM_SRI
  cout << " - using SRILM vocabulary with " << voc_size << " words\n";
#endif

  trainer = new TrainerNgramSlist(mach, &wlist, lm_fname, aux_dim);
  if (tgpos>=0) cout << " - the predicted word is at position " << tgpos << endl;
  trainer->BlockSetTgpos(tgpos);
 
  return true;
}

//
// Request the LM probs for all n-grams in one sentence
// The actual sentence log-proba will be calculated in FinishPending()
//
void NbestCSLM::RescoreHyp(Hypo &hyp, const int lm_pos)
{

  if (NULL == trainer)
    return;
  debug2("NbestCSLM::RescoreHyp(): lm_pos=%d, mode=%d\n", lm_pos, mode);
  WordID *wptr = NULL;
  int nw = trainer->GetSentenceIds(wptr, hyp.GetCstr(), mode & RESCORE_MODE_BOS, mode & RESCORE_MODE_EOS);

    // check whether we have enough space left to request all the n-grams from this hypo
    // (this needs to be done in one block since we will calculate the cumulated sentence proba)
  if (nw > trainer->BlockGetFree()) FinishPending();

    // allocate memory to store the delayed LM probabilities
  delayed_hyps.push_back(new HypSentProba(hyp, lm_pos, nw)); // (nw-1) would be actually enough
  debug2(" - allocate mem for %d words: addr=%p\n", nw, delayed_hyps.back()->GetAddrP());

   // request n-grams that are shorter then CSLM order, starting with 2-, 3-, ... n-gram
  int n=2;
  vector<REAL> aux_data_vec = (vector<REAL>) hyp.GetAuxData();
  int size = hyp.GetAuxDim();
  REAL* aux_data = new REAL[size];
  int j=0;
  for (vector<REAL>::iterator x = aux_data_vec.begin(); x != aux_data_vec.end(); x++) {
        aux_data[j]= *x;
        j++;
  }
  while (n<lm_order && n<=nw) {
    debug2(" - call BlockEval() for %dst %d-gram (short)\n", n-1, n);
    trainer->BlockEval(wptr, n, delayed_hyps.back()->GetAddrP()+n-2, aux_data);
    n++;
  }
    // request all remaining full n-grams
  while (n<=nw) {  // we have n-1 full n-grams in a sentence with n-words
    debug2(" - call BlockEval() for %dst %d-gram\n", n-1, lm_order);
    trainer->BlockEval(wptr, lm_order, delayed_hyps.back()->GetAddrP()+n-2, aux_data); // last address will be base+n-1
    n++, wptr++;
  }
  delete aux_data;
  return;
}


void NbestCSLM::FinishPending()
{
  debug1("NbestCSLM::FinishPending(): process %u delayed requests for complete hyps\n", (uint) delayed_hyps.size());
  trainer->BlockFinish();

  for (vector<HypSentProba*>::iterator i = delayed_hyps.begin(); i != delayed_hyps.end(); i++) {
    (*i)->SetSentProba();
    delete (*i);
  }
  delayed_hyps.clear();
}

//
//
//
float NbestCSLM::GetValue ()
{
  Error("NbestCSLM::GetValue() not implemented\n");
  return 0;
}

void NbestCSLM::Stats()
{
  trainer->BlockStats();
}

void NbestCSLM::RescoreNgrams (vector<string> &ngrams, REAL *probs, REAL *aux_data)
{
  if ((ngrams.size() == 0) || (NULL == trainer))
    return;
  WordID *wptr = NULL;

  for (size_t ni = 0 ; ni < ngrams.size() ; ni++)
      // split line into words and request CSLM proba
    if (trainer->GetSentenceIds(wptr, ngrams[ni], false, false) >= lm_order)
      trainer->BlockEval(wptr, lm_order, probs + ni, aux_data);

  trainer->BlockFinish();
}
