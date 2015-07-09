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

#ifndef _PtableMosesPtree_h
#define _PtableMosesPtree_h

using namespace std;

#include "Ptable.h"
#include "Hypo.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// from Moses:
#include <TranslationModel/PhraseDictionaryTree.h>
#include <Util.h>


// interface class to Moses binary on-disk prahse tables
// (implementation with a prefix tree)

const REAL PROBA_COPY_UNK (1);	// translation probability when an unknown word is copied from source to target
const REAL PROBA_NOT_IN_PTABLE (1e-20);	// translation probability when a phrase pair is not found in the Moses phrase table
					// this can happen when some words are mapped to <unk> because of limited source or target vocabularies

//
// helper class to store and compare Phrase requests
// ugly C-style structure, but this seems to be more efficient

/*
struct PhraseReq {
  Align	a;
  vector<string>  &trgw;
  int cnt;
  REAL *res_ptr;
};
*/

class PtableMosesPtree {
 private:
   vector<Moses::PhraseDictionaryTree*> ptree;	// main and eventually secondary phrase tables
   vector<int> pos_scores;			// starting position of the scores to be returned from each phrase table
   int nscores;					// number of scores to be returned (must be same for all phrase-tables)
   vector<Moses::StringTgtCand> tgtcands;
 public:
  PtableMosesPtree() {};
  virtual ~PtableMosesPtree();
  virtual void Read(const string &, const int, const char*);		// read next phrase table from file
  virtual REAL GetProb(vector<string>&, vector<string>&, vector<float> * =NULL);		// return one proba for a tokenized phrase-pair or vector of scores
  //virtual REAL GetProbWid(REAL *src, WordID *tgt) {return 0;} 
  virtual void RescoreHyp (Hypo&, vector<string> &, const int);
  virtual int GetNscores() {return nscores; }
};

#endif
