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

#ifndef _NBEST_H_
#define _NBEST_H_

using namespace std;

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "Toolsgz.h"
#include "Hypo.h"
#include "NbestLM.h"
#include "NbestCSTM.h"
#include "PtableMosesPtree.h"

#include "AlignReq.h"

class NBest {
  int 		   id;
  vector<string>   srcw;	// source sentence parsed into words (only available for TM rescoring)
  vector<Hypo> nbest;
  bool ParseLine(inputfilestream& inpf, inputfilestream& auxf, const int, const bool, const int);
    // Delayed translation model rescoring
  int max_req;			// max number of request cumulated before we perform them in a block
  int nreq;			// current number of request cumulated
  AlignReq *areq;		// array to allocate all requests
  int nb_diff_align;		// stats
 public:
  NBest(inputfilestream&, inputfilestream& , const int=0, const bool =false , const int=0);
  ~NBest();
  int NbNBest() {return nbest.size(); }
  int NbPhrases();
  int NbDiffPhrases() {return nb_diff_align; }
  void CalcGlobal(Weights&);
  void Sort(); // largest values first
  void Write(outputfilestream&, int=0);
  void AddID(const int offs);
  void RescoreLM(NbestLM&, const int); // recalc LM score on hypothesis (uses optional auxiliary data)
    // Delayed translation model rescoring with on disk phrase table
  void RescorePtable(PtableMosesPtree&, ifstream&, const int);
  void RescorePtableInv(PtableMosesPtree&, ifstream&, const int);
  void BlockFinish(PtableMosesPtree&, int);
  REAL GetAlignProb(PtableMosesPtree&, AlignReq&, const int, vector<float>* = NULL);
    // Delayed translation model rescoring with CSTM
  void RescorePtable(NbestCSTM&, ifstream&, const int);
  void RescorePtableInv(NbestCSTM&, ifstream&, const int);
  void BlockFinish(NbestCSTM&, int);
  void ForwAndCollect(int, int, int);
};


#endif
