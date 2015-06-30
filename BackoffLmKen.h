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

#ifndef _BackoffLmKen_h
#define _BackoffLmKen_h


#include <vector>

#include "BackoffLm.h"
#include "Tools.h"
#include "WordList.h"

// from KENLM
#include <lm/model.hh>
#include <lm/word_index.hh>

class BackoffLmKen : public BackoffLm {
 private:
  lm::ngram::Model            *ken_ngram;
  const lm::ngram::Vocabulary *ken_vocab;
  std::vector<LMWordIndex>    map_cslm2ken; // map internal CSLM indices to internal KENLM WordIndex
  std::vector<WordID>         map_ken2wid;  // map internal KENLM WordIndex to internal WordID
  std::vector<WordID>         wid_vect;     // vector of WordID in sentence

 public:
  BackoffLmKen(char *p_fname, int p_max_order, const WordList &wlist);
  virtual ~BackoffLmKen();

  /**
   * returns order of the loaded LM
   */
  inline virtual int GetOrder() {
    return ((NULL != ken_ngram) ? ken_ngram->Order() : 0); }

  /**
   * returns size of the vocabulary
   */
  inline virtual WordID GetVocSize() {
    return ((NULL != ken_vocab) ? (ken_vocab->Bound() + 1) : 0); }

  /**
   * gets WordID of words in sentence
   * @param wid output table of WordID (allocated internally)
   * @param sentence input sentence
   * @param bos start sentence with BOS
   * @param eos end sentence with EOS
   * @return number of words
   */
  virtual int GetSentenceIds(WordID *&wid, const std::string &sentence, bool bos, bool eos);

  /**
   * gets backoff LM P(w|ctxt) from sequence of words
   */
  inline virtual REAL BoffPw(char **ctxt, char *w, int req_order) {
    return exp(BoffLnPw(ctxt, w, req_order)); }

  /**
   * gets ln of backoff LM P(w|ctxt) from sequence of words
   */
  virtual REAL BoffLnPw(char **ctxt, char *w, int req_order);

  /**
   * gets backoff LM P(w|ctxt) from sequence of CSLM indices
   */
  inline virtual REAL BoffPid(REAL *ctxt, WordID predw, int req_order) {
    return exp(BoffLnPid(ctxt, predw, req_order)); }

  /**
   * gets ln of backoff LM P(w|ctxt) from sequence of CSLM indices
   */
  virtual REAL BoffLnPid(REAL *ctxt, WordID predw, int req_order);

  /**
   * gets ln of backoff LM P(w|ctxt) from sequence of CSLM indices, without mapping
   */
  virtual REAL BoffLnStd(WordID *ctxt, WordID predw, int req_order);
};

#endif
