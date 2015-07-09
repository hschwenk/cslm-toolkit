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


#ifndef _NBESTCSTM_H_
#define _NBESTCSTM_H_

using namespace std;

#include "Mach.h" // from the CSTM toolkit
#include "TrainerPhraseSlist.h" 
#include "WordList.h"

class NbestCSTM {
private:
  WordList src_wlist;
  WordList tgt_wlist;
  Mach *mach;
  TrainerPhraseSlist *trainer;
  bool stable_sort;	// use stable sort (default=true), set to false for compatibility with CSLM <= V3.0
public:
  NbestCSTM() : src_wlist(true), tgt_wlist(true), mach(NULL), trainer(NULL), stable_sort(true) {}
  virtual ~NbestCSTM();
  virtual void SetSortBehavior(bool val) {stable_sort=val;}
  virtual void Read (char*, char*, char* , char*, int, char*);
  virtual void AddToInput(int, vector<string> &, int, int);
  virtual void LookupTarget(vector<string> &v, WordID *);
  virtual void Stats() {trainer->BlockStats();}
  friend class NBest;
};

#endif
