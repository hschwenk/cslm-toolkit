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


#include <stdlib.h>     // exit()
#include "NbestLMSRI.h"
#include "Tools.h"

NbestLMSRI::NbestLMSRI()
 : sri_vocab(0), sri_ngram(0)
{
 //cerr << "NbestLMSRI::NbestLMSRI called" << endl;
}

NbestLMSRI::~NbestLMSRI() {
  delete sri_vocab;
  delete sri_ngram;
}

bool NbestLMSRI::Read (const string &fname, int const order) {

  //cout << " - reading SRI LM from file " << fname << endl;
  sri_vocab = new Vocab();
  sri_ngram = new Ngram(*sri_vocab, order);

    // reading LM
  lm_order = order;
  sri_ngram->setorder(lm_order);
  sri_ngram->skipOOVs() = false;
  File ngram_file(fname.c_str(), "r");
  sri_ngram->read(ngram_file, 0);

  sri_idx_unk = sri_vocab->unkIndex();
  sri_idx_bos = sri_vocab->ssIndex();
  sri_idx_eos = sri_vocab->seIndex();

    // get order and number of n-grams
  nb_ngrams.push_back(sri_vocab->numWords());
  cout << "   vocabulary: " << nb_ngrams[0] << " words; ngrams:";
  for (int o=1; o<=lm_order; o++) {
    nb_ngrams.push_back(sri_ngram->numNgrams(o));
    cout << " " << nb_ngrams.back();
    //if (nb_ngrams[o]==0) {order=o-1; break; };
  }
  cout << endl;
  
  return true;
}

//
//
//
void NbestLMSRI::RescoreHyp (Hypo &hyp, const int lm_pos, REAL*)
{
  debug2("NbestLMSRI::RescoreHyp(): lm_pos=%d, mode=%d\n", lm_pos, mode);
  static TextStats tstats;
  static const int max_words=16384;
  static const int max_chars=max_words*16;
  static char str[max_chars];
  static VocabString vstr[max_words+1];

  if (mode != (RESCORE_MODE_BOS | RESCORE_MODE_EOS)) {
    ErrorN("ERROR: mode is set to %d, but the SRILM automatically surrounds the sentence with <s> and </s>", mode);
  }

  strcpy(str,hyp.GetCstr()); // we need to copy since parseWords() modifies the string
  int nw = sri_vocab->parseWords(str, vstr, max_words + 1);
  if (nw == max_words+1) Error("too many words in one hypothesis\n");
  debug1(" parsing found %d words\n", nw);

  float logP = sri_ngram->sentenceProb(vstr, tstats);
  debug1("log10P=%e / 5d\n", logP);
  hyp.SetFeature(logP,lm_pos);
  return;
}

