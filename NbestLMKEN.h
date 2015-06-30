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

#ifndef _NBESTLMKEN_H_
#define _NBESTLMKEN_H_

#include <string>
#include <vector>
#include "NbestLM.h"

// from the KENLM toolkit
#include <lm/model.hh>

class NbestLMKEN : public NbestLM {
protected:
  const lm::ngram::Vocabulary *ken_vocab;	// KENLM vocabulary
  lm::ngram::Model            *ken_ngram;	// pointer on KENLM model
public:
  NbestLMKEN() : ken_vocab(0), ken_ngram(0) {};
  virtual ~NbestLMKEN();
  virtual bool Read (const string &, int const order = 4);
  virtual void RescoreHyp (Hypo &hyp, const int lm_pos, REAL* =NULL); // recalc log10 LM score on hypothesis
};

#endif
