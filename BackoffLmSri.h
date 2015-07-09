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

#ifndef _BackoffLmSri_h
#define _BackoffLmSri_h

#include <cstdio>
#include <vector>

#include "BackoffLm.h"
#include "Tools.h"

// from SRILM
#include <Vocab.h>
#include <Ngram.h>
#include "WordList.h"


class BackoffLmSri : public BackoffLm {
 private:
  static const int max_words=16384; // max words in a sentence
  Vocab            *sri_vocab;
  Ngram            *sri_ngram;
  int               sri_order;
  std::vector<VocabIndex> map_cslm2sri;		// map internal CSLM indices to internal SRI VocabIndex
  VocabIndex       *sri_context_idxs;		// internal storage of n-gram size
  WordID           wid_table[max_words];	// table of WordID in sentence
  void BackoffLmSri_init(char *p_fname, int p_max_order);

 public:
  BackoffLmSri(char *p_fname, int p_max_order)
 	: sri_vocab(NULL), sri_ngram(NULL), sri_order(0) {BackoffLmSri_init(p_fname, p_max_order); }
  BackoffLmSri(char *p_fname, int p_max_order, const WordList &wlist);
  virtual ~BackoffLmSri();
  inline virtual int GetOrder() { return sri_order; }
  inline virtual WordID GetVocSize() {
    return ((NULL != sri_vocab) ? sri_vocab->numWords() : 0); }

  /**
   * gets WordID of words in sentence
   * @param wid output table of WordID (allocated internally)
   * @param sentence input sentence
   * @param bos start sentence with BOS
   * @param eos end sentence with EOS
   * @return number of words
   */
  virtual int GetSentenceIds(WordID *&wid, const std::string &sentence, bool bos, bool eos);

  virtual REAL BoffPw(char **ctxt, char *w, int req_order)	// gets backoff LM P(w|ctxt) from sequence of words
    { Error ("BoffPw() not implmented for SRIL LMs"); return 0; }
  virtual REAL BoffLnPw(char **ctxt, char *w, int req_order)	// idem but ln of P(w|ctxt)
      // if the order of the back-off LM is smaller than we use the last n-1 words of the context
    { Error ("BoffLnPw() not implmented for SRIL LMs"); return -99; }
  virtual REAL BoffLnPid(REAL *ctxt, WordID predw, int req_order)
    // gets LOG_e backoff LM proba from a sequence of CSLM indices
    // if the order of the back-off LM is smaller than we use the last n-1 words of the context
    //   w1   w2  w3   ->  w4
    //            \ 2-gram /
    //        \-- 3-gram --/
    //   \---- 4-gram  ----/
  {
#ifdef DEBUG
    printf ("\nrequest SRI %d-gram: %d ", req_order, (WordID) ctxt[0]);
    for (int i=1; i<sri_order-1; i++) printf(", %d", (WordID) ctxt[i]);
    printf(" -> %d \n", predw);
#endif
    if (!sri_ngram) return NULL_LN_PROB;  // return constant value if we have no LM

      // SRILM requires a context vector which contains the words in REVERSE order
    int i;
    for (i=0; i<req_order-1; i++) {
      int j=sri_order-2-i;
      sri_context_idxs[i] = map_cslm2sri[(WordID) ctxt[j]]; // we need reverse order in context for SRI !!
      //printf(" - context position cslm=%d -> sri=%d, sri_idx=%d word=%s\n", j, i, sri_context_idxs[i], sri_vocab->getWord(sri_context_idxs[i]) );
    }
    sri_context_idxs[i]=Vocab_None; // terminate, this is needed to specify the length of the context
    //printf(" - predict cslm_id=%d, sri_idx=%d word=%s\n", predw, map_cslm2sri[predw], sri_vocab->getWord(map_cslm2sri[predw]) );
  
#ifdef DEBUG
    printf(" - SRI context: ");
    for (i=0; sri_context_idxs[i]!=Vocab_None; i++) {
       printf(" %s [%d]", sri_vocab->getWord(sri_context_idxs[i]), sri_context_idxs[i] );
    }
    printf(" -> %s [%d]", sri_vocab->getWord(map_cslm2sri[predw]), map_cslm2sri[predw]);
    printf (", log10P=%e\n", sri_ngram->wordProb(map_cslm2sri[predw], sri_context_idxs));
#endif

      // we need to convert from log_10 to ln
    return M_LN10 * sri_ngram->wordProb(map_cslm2sri[predw], sri_context_idxs);
  }
  virtual REAL BoffPid(REAL *ctxt, WordID predw, int req_order) {return exp(BoffLnPid(ctxt,predw,req_order)); }
  virtual REAL BoffLnStd(WordID *ctxt, WordID predw, int req_order)
  {
    // standard back-off n-gram wrapper,
    // SRILM properly shortens the context if we request an n-gram with an order that is larger then the back-off LM

    if (!sri_ngram) return NULL_LN_PROB;  // return constant value if we have no LM

    int i;
    for (i=0; i<req_order-1; i++) {	// build context vector in REVERSE order
      int j=req_order-i-2;
      sri_context_idxs[i] = ctxt[j];
      //debug4(" - context position cslm=%d -> sri=%d, sri_idx=%d word=%s\n", j, i, sri_context_idxs[i], sri_vocab->getWord(sri_context_idxs[i]) );
    }
    sri_context_idxs[i]=Vocab_None; // terminate, this is needed to specify the length of the context
    //debug3(" - predict cslm_id=%d, sri_idx=%d word=%s\n", predw, predw, sri_vocab->getWord(predw) );
  
#ifdef DEBUG
    printf(" - SRI %d-gram context: ",req_order);
    for (i=0; sri_context_idxs[i]!=Vocab_None; i++) {
       printf(" %s [%d]", sri_vocab->getWord(sri_context_idxs[i]), sri_context_idxs[i] );
    }
    printf(" -> %s [%d]", sri_vocab->getWord(predw), predw);
    printf (", log10P=%e\n", sri_ngram->wordProb(predw, sri_context_idxs));
#endif
    return M_LN10 * sri_ngram->wordProb(predw, sri_context_idxs); // convert from log_10 to ln
  }
};

#endif
