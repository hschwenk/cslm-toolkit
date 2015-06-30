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

#ifndef _TrainerNgramSlist_h
#define _TrainerNgramSlist_h

#include <ostream>
#include "Tools.h"
#include "Mach.h"
#include "ErrFct.h"
#include "DataNgramBin.h"
#include "TrainerNgram.h"
#include "WordList.h"
#include "BackoffLm.h"

//
// Class to train neural networks to predict n-gram probabilities
//  - we use a short list of words for which the NN predicts the proba
//  - the proba of the other words are obtained by a classical back-off LM
//  - the NN also predicts the proba mass of ALL the words not in the short slist
//    for this we use the last output neuron of the network


//
// helper class to store and compare one ngram LM request
// ugly C-style structure, but this seems to be more efficient

struct NgramReq {
  int ctxt_len;
  WordID *ctxt, wpred;
  int aux_len;
  REAL *aux;
  int bs;
  REAL *res_ptr;
};

/*
public:
  NgramReq(WordID *wid, int order, float *adrP)
    : ctxt_len(order-1), ctxt(new WordID[ctxt_len]), wpred(wid[ctxt_len]), res_ptr(adrP)
    { // printf("constructor NgramReq addr=%p\n", this); 
       for (int i=0; i<ctxt_len; i++) ctxt[i]=wid[i]; }
  ~NgramReq() { delete [] ctxt; }

   friend bool operator<(const NgramReq &n1, const NgramReq &n2)
   { // printf("compare %p[%d] < %p[%d]\n", n1, n1->ctxt[0], n2, n2->ctxt[0]);
     for (int i=0; i< (n1.ctxt_len < n2.ctxt_len) ? n1.ctxt_len : n2.ctxt_len; i++) {
       if (n1.ctxt[i] < n2.ctxt[i]) return true;
       if (n1.ctxt[i] > n2.ctxt[i]) return false;
     }
     return true; // both are equal
   }
   
   friend bool operator==(const NgramReq &n1, const NgramReq &n2)
   {  //printf("operator %p < %p\n", this, &n2);
     for (int i=0; i<n1.ctxt_len; i++) {
       if (n1.ctxt[i] != n2.ctxt[i]) return false;
     }
     return true; // both are equal
   }

   friend int NgramReqComp(const void *v1, const void *v2);

   void display() {
     printf(" %d-word ctxt:", ctxt_len);
     for (int c=0; c<ctxt_len; c++) printf(" %d", ctxt[c]);
     printf(" -> %d, addr=%p\n", wpred, res_ptr);
   }
};
*/


class TrainerNgramSlist : public TrainerNgram
{
private:
  int		mode;		// similar to class DataNgramBin, used to decide which n-grams
				// are processed during validation,
				// TODO: we should get this info automatically from the DataFile
				// During training we always present all n-grams to the NN
				// -> it is possible to select the mode by the DataFile
    // copies of important fields
  int		max_inp_idx;	// largest index -1 of a word at the input (# of entries in projection table)
  int		nb_ex_slist;	// total number of examples processed in slist
  int		nb_ex_short;	// total number of examples with short n-grams
  char		*lm_fname;
// TODO: use WordID vector for targets in order to make less casts
  WordID	*lm_buf_target;	// keep track of word indices not in short list (which are all done by same output)
  WordID	slist_len;	// length of slist (this is set to the size of the output layer MINUS ONE)
  int tgpos;		// position of target word in n-gram
  BackoffLm	*blm;		// this must be a pointer so that polymorphism will work !
    // CSLM use different indices of the words than the provided word list
    // they are mapped so that the most frequent words have consecutive indices
    // This speeds up calculation and facilitates the decision which words are in the short list
  WordList	*wlist;
#ifdef DEBUG
  vector<char*>  words;		// give UTF8 word for a given CSLM internal index
#endif
  REAL DoTestDev(char*, bool);	// internal helper function
  void DoConstructorWork();	// internal helper function for the various constructors
    // data and functions for block processing
  int	max_req;		// max number of request cumulated before we perform them in a block
  int	nreq;			// current number of request cumulated before we perform them in a block
  NgramReq *req;		// array to allocate all requests
  int	nb_ngram;		// total number of n-grams requested
  int	nb_forw;		// stats on total number of forward passes
  void FreeReq();
protected:
  virtual void InfoPost();			// dump information after finishing a training epoch
  virtual void ForwAndCollect(int,int,int,bool);	// internal helper function
public:
  TrainerNgramSlist(Mach*, Lrate*, ErrFct*,	// mach, lrate, errfct
	  const char*, const char*, const char*,	// train, dev, LM
	  REAL =0, int =10, int =0);			// wdecay, max epochs, current epoch
  TrainerNgramSlist(Mach*, ErrFct*, Data*, char*);	// for testing only: mach, errfct, binary data, LM
  TrainerNgramSlist(Mach*, WordList*, char*, int=0);	// for general proba calculation: mach, word list, LM, auxiliary data dimension
  virtual ~TrainerNgramSlist();
  virtual REAL Train();				// train for one epoch
  virtual REAL TestDev(char* pfname=NULL)	// test current network on dev data and save outputs into file
    { return DoTestDev(pfname, false); }
  virtual REAL TestDevRenorm(char* pfname=NULL)	// same, but renormalize all probabilities with back-off LM proba-mass (costly)
    { return DoTestDev(pfname, true); }
  inline virtual WordID GetVocSize()  // returns size of the vocabulary
    { return ((NULL != blm) ? blm->GetVocSize() : 0); }
  inline virtual int GetSentenceIds(WordID *&wid, const std::string &sentence, bool bos, bool eos) // gets WordID of words in sentence
    { return ((NULL != blm) ? blm->GetSentenceIds(wid, sentence, bos, eos) : 0); }                 // (returns number of words)

    // fast block evaluation functions
  virtual void BlockSetTgpos(const int tp)		// sets the tgpos in the n-gram
    { if(tp==-1) { tgpos=iaux; return; }
      if (tp<0 || tp>iaux) ErrorN("target position must be in [0,%d], use -1 for the last word\n",iaux);
      tgpos=tp;
    }
  virtual void BlockSetMax(int=65536);			// sets size of requests to be delayed before actual calculation
  virtual int BlockGetFree()				// returns the number of requested that can still be delayed before
    { return (int) max_req-nreq; }			// we evaluate a whole block. This can be used to keep together a sequence
							// of request, i.e. individual n-grams of a whole sentence
  virtual void BlockEval(WordID *wid, int o, REAL*p, REAL* =NULL);	// request n-gram, result WILL be stored at address (uses optional auxiliary data)
  virtual void BlockFinish();				// finishes all pending requests
  virtual void BlockStats();				// displays some stats on Block mode
};

#endif
