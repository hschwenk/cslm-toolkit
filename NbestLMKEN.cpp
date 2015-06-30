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


#include "NbestLMKEN.h"
using namespace std;
using namespace lm::ngram;

NbestLMKEN::~NbestLMKEN() {
  if (NULL != ken_ngram)
    delete ken_ngram;
}

bool NbestLMKEN::Read (const string &fname, int const)
{
  ken_ngram = new ProbingModel(fname.c_str());
  if (NULL == ken_ngram) {
    cout << "   error" << endl;
    ken_vocab = NULL;
    return false;
  }
  else {
    ken_vocab = &(ken_ngram->GetVocabulary());
    nb_ngrams.push_back(ken_vocab->Bound() + 1);
    cout << "   vocabulary: " << nb_ngrams[0] << " words" << endl;
    return true;
  }
}

//
//
//
void NbestLMKEN::RescoreHyp (Hypo &hyp, const int lm_pos, REAL*)
{
  float logP = 0;
  if (NULL != ken_ngram) {
    State state((mode & RESCORE_MODE_BOS) ? ken_ngram->BeginSentenceState() : ken_ngram->NullContextState()), out_state;
    istringstream iss(hyp.GetCstr());
    while (iss) {
      string s;
      iss >> s;
      if (!s.empty()) {
        logP += ken_ngram->Score(state, ken_vocab->Index(s), out_state);
        state = out_state;
      }
    }
    if (mode & RESCORE_MODE_EOS)
      logP += ken_ngram->Score(state, ken_vocab->EndSentence(), out_state);
  }
  hyp.SetFeature(logP, lm_pos);
}

