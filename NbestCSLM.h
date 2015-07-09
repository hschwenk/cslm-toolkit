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


#ifndef _NBESTCSLM_H_
#define _NBESTCSLM_H_

using namespace std;

#include "NbestLM.h"
#include "Mach.h" // from the CSLM toolkit
#include "TrainerNgramSlist.h" 
#include "Tools.h"
#include "WordList.h"

class HypSentProba {
private:
  Hypo	&hyp;		// hypothesis for which we want to modify the sentence probability
  int   lm_pos;
  int	nw;		// number of words in this sentences
  REAL	*p;		// array to store the n-gram log probabilities
public:
  HypSentProba(Hypo &p_hyp, int p_pos, int p_nw) : hyp(p_hyp), lm_pos(p_pos), nw(p_nw), p(new REAL[nw]) {
    debug1("HypSentProba(): alloc addr %p\n", p);
  };
  ~HypSentProba() { if(p) delete [] p; }
  REAL *GetAddrP() {return p;}
  void SetSentProba()
  {
    REAL logP=0;
    for (int i=0; i<nw-1; i++) {
      debug2("HypSentProba(): logp=%e at pos %d\n", p[i],i);
      logP+=p[i];
    }
    debug3("           =>   store sentence logP=%e (log10=%e) at pos %d\n", logP,logP/M_LN10,lm_pos);
    hyp.SetFeature(logP,lm_pos);
  }
};
  
  
class NbestCSLM : public NbestLM {
protected:
  WordList wlist;
  Mach *mach;
    // storage to cumulate delayed LM probabilities for several hypotheses
  TrainerNgramSlist *trainer;
  vector<HypSentProba*> delayed_hyps;
public:
  NbestCSLM() : mach(NULL), trainer(NULL) {delayed_hyps.clear(); }
  virtual ~NbestCSLM();
  virtual float GetValue();
  virtual bool Read (char*, char*, char*, int=-1, int=0);
  virtual void RescoreHyp (Hypo &hyp, const int lm_pos); // recalc LM score on hypothesis, returns log10 probability
  virtual void RescoreNgrams (vector<string> &, REAL*, REAL* =NULL); // calc CSLM score for a vector of n-grams (uses optional auxiliary data)
  virtual void FinishPending();
  virtual void Stats();
};

#endif
