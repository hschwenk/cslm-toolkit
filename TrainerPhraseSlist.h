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

#ifndef _TrainerPhraseSlist_h
#define _TrainerPhraseSlist_h

#include <ostream>
#include "Tools.h"
#include "Mach.h"
#include "ErrFct.h"
#include "DataPhraseBin.h"
#include "Trainer.h"
#include "WordList.h"

#include "PtableMosesPtree.h"
#include "AlignReq.h"

//
// Class to train neural networks to predict phrase probabilities
//  - we use a short list of target words for which the NN predicts the proba
//  - the proba of the other target words are obtained by a classical Moses phrase table
//  - the NN also predicts the proba mass of ALL the words not in the short slist
//    for this we use the last output neuron of the network


class TrainerPhraseSlist : public Trainer
{
private:
  int		max_inp_idx;		// largest index -1 of a word at the input (# of entries in projection table)
  int		tg_nbphr;		// number of phrases at output, odim should be (tg_slist_len+1) * tg_nbphr
  int		dim_per_phrase;		// output dimension of each phrase prediction layer (must be equal size)
  WordID	tg_slist_len;		// length of slist (this is set to dim_per_phrase MINUS ONE)
  WordList	*sr_wlist;
  WordList	*tg_wlist;
  vector<Mach*> phrase_mach;		// pointer to the output machine for each phrase
  vector<ErrFct*> mach_errfct;		// each individual machine has its own error function with local memory
					// in this version of the Trainer the error function is identical to all machines
					// (we use the one in the local variable of the mother class Trainer)
 
  PtableMosesPtree	*ptable;	// classical phrase table

    // handling of short sequences
    // 			input		output	
    // NULL_WORD	set proj=0	set grad=0
    // EOS		as normal word	as normal word
    //
  WordID eos_src, eos_tgt;		// defaults to NULL_WORD if no special symbol in word list

    // various stats
  int		nb_ex_slist;		// total number of examples processed in slist
  int		nb_ex_short_inp;	// total number of incomplete input phrases
  int		nb_ex_short_tgt;	// total number of incomplete target phrases
  int		nb_tg_words;		// total number of target words (there can be several target words for a phrase pair)
  int		nb_tg_words_slist;	// total number of target words which are in short list
// TODO: use WordID vector for targets in order to make less casts 
  WordID	*buf_target_wid;	// used instead of buf_target to avoid casts between REAL and WordID
					// size is odim x bsize
  WordID	*buf_target_ext;	// similar to buf_target_wid[], but keep even word id out side of short list
					// needed to request probas from external phrase table
  REAL		*buf_target_in_blocks;	// same data than in buf_target of Trainer class, but re-arranged in blocks for individual machines
#ifdef BLAS_CUDA
  vector<REAL*> gpu_target;	// copied from trainer to GPU
#endif
#ifdef DEBUG
  vector<char*>  words;			// give UTF8 word for a given CSLM internal index
#endif
  REAL DoTestDev(char*, bool);	// internal helper function
  void DoConstructorWork();	// internal helper function for the various constructors
    // data and functions for block processing
  int	nb_forw;		// stats on total number of forward passes
  void GetMostLikelyTranslations(ofstream&,REAL*,int);
protected:
  virtual void InfoPost();			// dump information after finishing a training epoch
public:
  TrainerPhraseSlist(Mach*, Lrate*, ErrFct*,	// mach, lrate, errfct
	  const char*, const char*, const char*, int,	// train, dev, external phrase table, number of scores
	  REAL =0, int =10, int =0);			// wdecay, max epochs, current epoch
  TrainerPhraseSlist(Mach*, ErrFct*, Data*,	// for testing only: mach, errfct, binary data
	  char*, int);				// external phrase table, number of scores
  TrainerPhraseSlist(Mach*, WordList*, WordList*,	// for general proba calculation: mach, src word list, tgt word list
	  char*, int , char*);			// external phrase table, number of scores, score specif
  virtual ~TrainerPhraseSlist();
  virtual REAL Train();				// train for one epoch
  virtual REAL TestDev(char* =NULL);		// test current network on dev data and save outputs into file
    // fast block evaluation functions
  virtual void StoreInput(int b, int d, REAL val) {buf_input[b*bsize+d]=val;}
  virtual void ForwAndCollect(vector< vector<string> > &, AlignReq*, int,int,int,int);	// for nbest rescoring
  virtual void BlockStats();				// display some stats on Block mode
    // interface functions
  virtual int GetTgtNbPhr() {return tg_nbphr; }
  virtual int GetSlistLen() {return tg_slist_len; }
  virtual REAL *GetBufInput() {return buf_input; }
};

#endif
