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

using namespace std;

#include <string>

#include "Tools.h"
#include "Hypo.h"
#include "NbestCSTM.h"
#include "ErrFctSoftmCrossEntNgramMulti.h"


NbestCSTM::~NbestCSTM() {
  if (mach) delete mach;
  if (trainer) delete trainer;
}


void NbestCSTM::Read(char *fname, char *wl_src_fname, char *wl_tgt_fname, char *pt_fname, int nscores, char *scores_specif)
{
  ifstream ifs;
  ifs.open(fname,ios::binary);
  CHECK_FILE(ifs,fname);
  mach = Mach::Read(ifs);
  ifs.close();

  mach->Info();

  // create vocabulary from our source word list, this must be exactly the same order than in extract2bin !!!
  cout << " - reading source word list from file " << wl_src_fname << flush;
  src_wlist.SetSortBehavior(this->stable_sort);
  src_wlist.Read(wl_src_fname);
  cout << ", got " << src_wlist.GetSize() << " words" << endl;

  // create vocabulary from our target word list, this must be exactly the same order than in extract2bin !!!
  cout << " - reading target word list from file " << wl_tgt_fname << flush;
  tgt_wlist.SetSortBehavior(this->stable_sort);
  tgt_wlist.Read(wl_tgt_fname);
  cout << ", got " << tgt_wlist.GetSize() << " words" << endl;

  trainer = new TrainerPhraseSlist(mach, &src_wlist, &tgt_wlist, pt_fname, nscores, scores_specif);
}

void NbestCSTM::AddToInput(int b, vector<string> &vsrcw, int sb, int se)
{
  int idim=mach->GetIdim();
  if (sb-se+1 > idim) {
    ErrorN("NbestCSTM::AddToInput(): source phrase too long (%d) for machine (%d)\n", sb-se+1, idim);
  }

  REAL *iptr=trainer->GetBufInput() + b*idim;
  int i=0;

  // get index of each source word
  debug0("NbestCSTM::AddToInput():");
  REAL unk_wi = (REAL) src_wlist.GetIndex(WordList::WordUnknown);
  for (int w=sb; w<=se; w++) {
    WordList::WordIndex wi = src_wlist.GetIndex(vsrcw[w].c_str());
    if (wi==WordList::BadIndex) {
      fprintf(stderr, "ERROR: source word not found: %s\n", vsrcw[w].c_str());
      *iptr++ = unk_wi;
    }
    else 
      *iptr++ = (REAL) wi;
    debug2(" %s->%f", vsrcw[w].c_str(), iptr[-1]);
    i++;
  }
  debug0("\n");

  // fill up input phrase to the dimension of the machine
  for (; i<idim; i++) *iptr++=NULL_WORD;
}
 
void NbestCSTM::LookupTarget(vector<string> &vtrgw, WordID *wid)
{
  int nph=trainer->GetTgtNbPhr();
  int vdim=(int) vtrgw.size();

  if (vdim>nph) {
    ErrorN("NbestCSTM::MapTarget(): phrase (%d) exceeds length of machine (%d)\n",vdim, nph);
  }
  
  int i;
  debug0("NbestCSTM::LookupTarget():");
  for (i=0; i<vdim; i++) {
    WordList::WordIndex wi = tgt_wlist.GetIndex(vtrgw[i].c_str());
    if (wi==WordList::BadIndex) {
      //ErrorN("ERROR: target word not found: %s\n", vtrgw[i].c_str());
      // TODO: count these events
      
      // this has as effect that the word won't be processed by the CSTM (out of short list)
      // maybe the external phrase table knows it?
      wid[i]=trainer->GetSlistLen();
    }
    else
      wid[i] = (WordID) wi;
    debug2(" %s->%d", vtrgw[i].c_str(), wid[i]);
  }
  debug0("\n");

  // fill up
  for (; i<nph; i++) wid[i] = NULL_WORD;
}
