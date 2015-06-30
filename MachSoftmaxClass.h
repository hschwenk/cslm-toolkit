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
 *
 * Machine that computes classification probabilities for words in a vocabulary
 * of size n, divided in c classes. The probabilities are computed as the probability
 * of a class c and the conditional probability of a word w given a class, given the
 * context h:
 * P(w|h) = P(c|h) * P(w|c,h)
 * Each of these probabilities is computed as a softmax:
 *   a_i = exp(a_i) / sum_k a_k
 * with a_k is the kth output of a linear machine
 *
 * There is one softmax for the probabilities of classes, and one for each of
 * the classes (that computes the probabilities of words in that class).
 *
 * This enables us to compute the log-likelihood of one word without having to
 * compute the probabilities for all words: we only need the probabilities of
 * all classes, and the conditional probability of all words in that class.
 */

#ifndef _MachSoftmaxClass_h
#define _MachSoftmaxClass_h

#include "Mach.h"
#include "MachLin.h"
#include "MachSoftmax.h"
#include "WordList.h"

class MachSoftmaxClass : public MachLin
{
protected:
  // A MachSoftmax that predicts the class, to encapsulate that part.
  Mach *class_softm_mach; //could be MachSoftmax or MachSoftmaxStable
  // The word list, that contains the information defining the architecture
  WordList *wlist;
  int n_classes;
  int max_class_size;
  // Buffer containing the offset and lengths of the target class in the vocabulary
  int* target_class_info;
  int stable;
  // Read and write binary data
  virtual void ReadData(istream&, size_t, int=0);
  virtual void WriteData(ostream&);
  // create a copy of the machine, sharing the parameters
  MachSoftmaxClass(const MachSoftmaxClass &);
public:
  MachSoftmaxClass(const int p_idim=0, const int p_odim=0, const int p_bsize=128,
                   const ulong p_nbfw=0, const ulong p_nbbw=0, const int stable=1);
  virtual ~MachSoftmaxClass();
  virtual MachSoftmaxClass *Clone() {return new MachSoftmaxClass(*this);}	// create a copy of the machine
  virtual int GetMType() {return file_header_mtype_softmax_class;};	// get type of machine
  // get number of allocated parameters
  virtual ulong GetNbParams();
  virtual void Info(bool=false, char *txt=(char*)"");	// display (detailed) information on machine
  virtual bool CopyParams(Mach* mach);	// copy parameters from another machine
  virtual void SetUp(WordList* p_wlist);	// Sets machine architecture
  virtual void Forw(int=0, bool=false);	// calculate outputs for current inputs
    // backprop gradients from output to input and update all weights
  virtual void Backw (const float lrate, const float wdecay, int=0);
  virtual REAL* GetDataOutClass();
  virtual void SetGradOutClass(REAL* data);
  virtual void SetTargetInfo(int* info);
};

#endif

