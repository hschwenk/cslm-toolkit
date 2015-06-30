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

#ifndef _NBESTLM_H_
#define _NBESTLM_H_

using namespace std;

#include <string>
#include <vector>
#include "Hypo.h"

#define RESCORE_MODE_BOS 1
#define RESCORE_MODE_EOS 2

class NbestLM {
protected:
  string fname;  // translation
  int lm_order;  // order of NbestLM
  int mode;
  bool stable_sort;	// use stable sort (default=true), set to false for compatibility with CSLM <= V3.0
  vector<int> nb_ngrams;  // nb of ngrams per order, nb_ngrams[0] is voc. size
public:
  NbestLM() : mode(RESCORE_MODE_BOS | RESCORE_MODE_EOS), stable_sort(true) {};
  virtual ~NbestLM() {};
  virtual float GetValue() {return 0; };
  virtual void SetSortBehavior(bool val) {stable_sort=val;}
  virtual bool Read (const string &, int const order = 4);
  virtual void RescoreHyp (Hypo &hyp, const int lm_pos) {}; // recalc LM score on hypothesis
  virtual void FinishPending() {};	// finish pending requests, only used for CSLM
  virtual void Stats() {};		// display stats
};

#endif
