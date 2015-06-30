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

#ifndef _BackoffLm_h
#define _BackoffLm_h

#include <string>
#include "Tools.h" // for type WordID

// We must be very careful with the indices
//  - most LM toolkits have their own internal word list
//  - binary ngram data files us indices with respect to their word list
//    (ideally, this word list should be identical to the one of the LM!)
//  - the CSLM code with short list performs a mapping of the binary indices
//    of the datafiles according to the 1-gram frequency
//
//

#define NULL_LN_PROB (1.0)   // this value must not be possible as a normal return value of ln Prob

class BackoffLm {
 private:
 public:
  BackoffLm() {};
  virtual ~BackoffLm() {};
  inline virtual int GetOrder() {return 0; };	// returns order of the loaded LM
  inline virtual WordID GetVocSize() {return 0; };  // returns size of the vocabulary
  virtual int GetSentenceIds(WordID *&wid, const std::string &sentence, bool bos, bool eos) {return 0; }; // gets WordID of words in sentence
  virtual REAL BoffPw(char **ctxt, char *w, int req_order) {return 0;}		// gets backoff LM P(w|ctxt) from sequence of words
  virtual REAL BoffLnPw(char **ctxt, char *w, int req_order) {return -99;}	// idem but ln of P(w|ctxt)
  virtual REAL BoffPid(REAL *ctxt, WordID predw, int req_order) {return 0;} 	// similar for sequences of CSLM indices
  virtual REAL BoffLnPid(REAL *ctxt, WordID predw, int req_order) {return -99;} 
  virtual REAL BoffLnStd(WordID *ctxt, WordID predw, int req_order) {return -99; } // simple wrapper w/o mapping
										   // req-order can be any value
};

#endif
