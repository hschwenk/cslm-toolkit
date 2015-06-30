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

#ifndef _NBESTLMSRI_H_
#define _NBESTLMSRI_H_

using namespace std;

#include <string>
#include <vector>
#include "NbestLM.h"

// from the SRI toolkit
#include <Vocab.h>
#include <Ngram.h>


class NbestLMSRI : public NbestLM {
protected:
  Vocab *sri_vocab;	// SRI vocabulary
  Ngram *sri_ngram;	// pointer on SRI model
  VocabIndex sri_idx_unk, sri_idx_bos, sri_idx_eos;
public:
  NbestLMSRI(); // : sri_vocab(0), sri_ngram(0) {};
  virtual ~NbestLMSRI();
  virtual float GetValue() {return 0; };
  virtual bool Read (const string &, int const order = 4);
  virtual void RescoreHyp (Hypo &hyp, const int lm_pos, REAL* =NULL); // recalc log10 LM score on hypothesis
};

#endif
