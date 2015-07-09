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

#ifndef _Ptable_h
#define _Ptable_h

using namespace std;

#include <string>
#include <vector>
#include "Tools.h"		// for type REAL
//#include "DataNgramBin.h"	// for type WordID

// interface class to classical phrase tables
//
//

#define NULL_LN_PROB (1.0)   // this value must not be possible as a normal return value of ln Prob

class Ptable {
 private:
 public:
  Ptable(const int, const int=2, const bool=false) {};				// initialize
  virtual ~Ptable() {};
  virtual void Read(const string &) {};						// read form file
  virtual REAL GetProb(vector<string>&, vector<string>&) {return 0;}		// get backoff LM P(w|ctxt) from seqeuence of words
  //virtual REAL GetProbWid(REAL *src, WordID *tgt) {return 0;} 
};

#endif
