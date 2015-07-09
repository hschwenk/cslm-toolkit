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

#include <cstdio>
#include <iostream>
#include "BackoffLmKen.h"
using namespace std;
using namespace lm::ngram;

BackoffLmKen::BackoffLmKen(char *p_fname, int, const WordList &wlist)
{
  if ((p_fname == NULL) || (p_fname[0] == '\0')) {
    // no back-off file
    ken_ngram = NULL;
    ken_vocab = NULL;
    return;
  }

  cout << " - reading back-off KENLM from file '" << p_fname << "'" << endl;
  ken_ngram = new ProbingModel(p_fname);
  if (NULL == ken_ngram) {
    cout << "   error" << endl;
    ken_vocab = NULL;
    return;
  }

  ken_vocab = &(ken_ngram->GetVocabulary());
  LMWordIndex ken_size = (ken_vocab->Bound() + 1);
  printf("   found %d-gram with vocabulary of %d words\n", (int) ken_ngram->Order(), ken_size);

    // set up mapping from/to KENLM indices
  WordList::WordIndex wlist_size = wlist.GetSize();
  map_cslm2ken.reserve(wlist_size);
  map_cslm2ken.resize(wlist_size);
  map_ken2wid.reserve(ken_size);
  map_ken2wid.resize(ken_size);
  WordList::const_iterator iter = wlist.Begin(), end = wlist.End();
  for (size_t ci = 0 ; iter != end ; iter++, ci++) {
    LMWordIndex wi = ken_vocab->Index(iter->word);
    map_cslm2ken[ci] = wi;
    if (wi == ken_vocab->NotFound())
      fprintf(stderr,"word %s not found at pos %zu\n", iter->word, ci);
    else
      map_ken2wid[wi] = iter->id;
  }
}

BackoffLmKen::~BackoffLmKen()
{
  if (NULL != ken_ngram)
    delete ken_ngram;
  map_cslm2ken.clear();
  wid_vect.clear();
}

/**
 * gets WordID of words in sentence
 * @param wid output table of WordID (allocated internally)
 * @param sentence input sentence
 * @param bos start sentence with BOS
 * @param eos end sentence with EOS
 * @return number of words
 */
int BackoffLmKen::GetSentenceIds(WordID *&wid, const string &sentence, bool bos, bool eos)
{
  if (NULL == ken_vocab)
    return 0;

  int nw = 0;
  wid_vect.clear();

    // start sentence with BOS ?
  if (bos) {
    wid_vect.push_back(map_ken2wid[ken_vocab->BeginSentence()]);
    nw++;
  }

  istringstream iss(sentence);
  while (iss) {
    string s;
    iss >> s;
    if (!s.empty()) {
      wid_vect.push_back(map_ken2wid[ken_vocab->Index(s)]);
      nw++;
    }
  }
  debug1(" parsing found %d words\n", nw);

    // end sentence with EOS ?
  if (eos) {
    wid_vect.push_back(map_ken2wid[ken_vocab->EndSentence()]);
    nw++;
  }

  wid = &(wid_vect.front());
  debug4("* split sent with %d words into %d-grams (bos=%d, eos=%d):\n", nw, ken_ngram->Order(), map_ken2wid[ken_vocab->BeginSentence()], map_ken2wid[ken_vocab->EndSentence()]);
  return nw;
}

/**
 * gets ln of backoff LM P(w|ctxt) from sequence of words
 */
REAL BackoffLmKen::BoffLnPw(char **ctxt, char *w, int req_order)
  // gets LOG_e backoff LM proba from a sequence of CSLM indices
  // if the order of the back-off LM is smaller than we use the last n-1 words of the context
  //   w1   w2  w3   ->  w4
  //            \ 2-gram /
  //        \-- 3-gram --/
  //   \---- 4-gram  ----/
{
#ifdef DEBUG
  printf ("\nrequest KENLM %d-gram: %s ", req_order, ctxt[0]);
  for (int i = 1; i < (req_order - 1); i++) printf(", %s", ctxt[i]);
  printf(" -> %s \n", w);
#endif
  if (NULL == ken_ngram)
    // return constant value if we have no LM
    return NULL_LN_PROB;

  State state(ken_ngram->NullContextState()), out_state;
  for (int i = 0; i < (req_order - 1); i++) {
    ken_ngram->Score(state, ken_vocab->Index(ctxt[i]), out_state);
    state = out_state;
    debug2(" - context position ken=%d, ken_idx=%d\n", i, ken_vocab->Index(ctxt[i]));
  }
  debug2(" - predict ken_idx=%d, log10P=%e\n", ken_vocab->Index(w), ken_ngram->Score(state, ken_vocab->Index(w), out_state));

    // we need to convert from log_10 to ln
  return M_LN10 * ken_ngram->Score(state, ken_vocab->Index(w), out_state);
}

/**
 * gets ln of backoff LM P(w|ctxt) from sequence of CSLM indices
 */
REAL BackoffLmKen::BoffLnPid(REAL *ctxt, WordID predw, int req_order)
  // gets LOG_e backoff LM proba from a sequence of CSLM indices
  // if the order of the back-off LM is smaller than we use the last n-1 words of the context
  //   w1   w2  w3   ->  w4
  //            \ 2-gram /
  //        \-- 3-gram --/
  //   \---- 4-gram  ----/
{
#ifdef DEBUG
  printf ("\nrequest KENLM %d-gram: %d ", req_order, (WordID) ctxt[0]);
  for (int i = 1; i < (req_order - 1); i++) printf(", %d", (WordID) ctxt[i]);
  printf(" -> %d \n", predw);
#endif
  if (NULL == ken_ngram)
    // return constant value if we have no LM
    return NULL_LN_PROB;

  State state(ken_ngram->NullContextState()), out_state;
  for (int i = 0; i < (req_order - 1); i++) {
    ken_ngram->Score(state, map_cslm2ken[(WordID) ctxt[i]], out_state);
    state = out_state;
    debug2(" - context position ken=%d, ken_idx=%d\n", i, map_cslm2ken[(WordID) ctxt[i]]);
  }
  debug3(" - predict cslm_id=%d, ken_idx=%d, log10P=%e\n", predw, map_cslm2ken[predw], ken_ngram->Score(state, map_cslm2ken[predw], out_state));

    // we need to convert from log_10 to ln
  return M_LN10 * ken_ngram->Score(state, map_cslm2ken[predw], out_state);
}

/**
 * gets ln of backoff LM P(w|ctxt) from sequence of CSLM indices, without mapping
 */
REAL BackoffLmKen::BoffLnStd(WordID *ctxt, WordID predw, int req_order)
  // gets LOG_e backoff LM proba from a sequence of CSLM indices
  // if the order of the back-off LM is smaller than we use the last n-1 words of the context
  //   w1   w2  w3   ->  w4
  //            \ 2-gram /
  //        \-- 3-gram --/
  //   \---- 4-gram  ----/
{
#ifdef DEBUG
  printf ("\nrequest KENLM %d-gram: %d ", req_order, ctxt[0]);
  for (int i = 1; i < (req_order - 1); i++) printf(", %d", ctxt[i]);
  printf(" -> %d \n", predw);
#endif
  if (NULL == ken_ngram)
    // return constant value if we have no LM
    return NULL_LN_PROB;

  State state(ken_ngram->NullContextState()), out_state;
  for (int i = 0; i < (req_order - 1); i++) {
    ken_ngram->Score(state, ctxt[i], out_state);
    state = out_state;
    debug2(" - context position ken=%d, ken_idx=%d\n", i, ctxt[i]);
  }
  debug3(" - predict cslm_id=%d, ken_idx=%d, log10P=%e\n", predw, predw, ken_ngram->Score(state, predw, out_state));

    // we need to convert from log_10 to ln
  return M_LN10 * ken_ngram->Score(state, predw, out_state);
}
