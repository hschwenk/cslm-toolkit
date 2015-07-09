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

#ifndef _ALIGNREQ_H_
#define _ALIGNREQ_H_

using namespace std;

#include <vector>
#include "Hypo.h"

struct AlignReq {
  int sb, se;		// requested alignment, we can use the word indices only since the source is constant for all hyps
  vector<string> tgph;	// target phrase 	
  WordID tgwid[16];	// mpped target wordID; TODO: this is an hack, we map many times the same target phrase
  Hypo *hyp;		// corresponding hypothesis
  int bs;		// index into bunch that will be processed by NN
  float *logP;	 	// log proba (may be several scores)
};

#endif
