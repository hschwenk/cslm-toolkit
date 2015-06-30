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
#include <iostream>

#include "Tools.h"
#include "MachSeq.h"

MachSeq::MachSeq()
 : MachMulti()
{
}

MachSeq::MachSeq(const MachSeq &m)
 : MachMulti(m)
{
}

MachSeq::~MachSeq()
{
  data_out=grad_in=NULL;  // prevent delete[] by ~Mach()
}

MachSeq *MachSeq::Clone()
{
  MachSeq *m = new MachSeq(*this);
  if (m != NULL)
    m->CloneSubmachs(*this);
  return m;
}

// set pointer of input data
void MachSeq::SetDataIn(REAL *data)
{
  data_in=data;
  if (machs.size() > 0) machs[0]->SetDataIn(data);
}
 
// set pointer of output gradient
void MachSeq::SetGradOut(REAL *data)
{
  grad_out=data;
  if (machs.size() > 0) machs.back()->SetGradOut(data);
}

void MachSeq::MachAdd(Mach *new_mach)
{
  if (machs.empty()) {
    machs.push_back(new_mach);
	// think about freeing memory
    idim=new_mach->GetIdim();
    bsize=new_mach->GetBsize();
    data_in=new_mach->GetDataIn();
    grad_in=new_mach->GetGradIn();
  }
  else {
    Mach *last_mach=machs.back();
    if (last_mach->GetOdim()!=new_mach->GetIdim()) {
      cout << "Current sequential machine:" << endl; Info(false);
      cout << "Newly added machine:" << endl; new_mach->Info(false);
      Error("input dimension of new sequential machine does not match");
    }
    if (bsize!=new_mach->GetBsize()) {
      cout << "Current sequential machine:" << endl; Info(false);
      cout << "Newly added machine:" << endl; new_mach->Info(false);
      Error("bunch size of new sequential machine does not match");
    }
    machs.push_back(new_mach);
 
      // connect new last machine to the previous one
    new_mach->SetDataIn(last_mach->GetDataOut());
    last_mach->SetGradOut(new_mach->GetGradIn());
  }

  activ_forw.push_back(true);
  activ_backw.push_back(true);

    // connect last machine to the outside world
  odim=new_mach->GetOdim();
  data_out=new_mach->GetDataOut();
  grad_out=new_mach->GetGradOut();
}

Mach *MachSeq::MachDel()
{
  if (machs.empty()) {
    Error("impossible to delete element from sequential machine: is already empty");
  }
  
  Mach *del_mach=machs.back();
  machs.pop_back();

  if (machs.empty()) {
    idim=odim=bsize=0;
    data_in=data_out=grad_in=grad_out=NULL;
  }
  else {
    Mach *last_mach=machs.back();

      // connect new last machine to the outside world
    odim=last_mach->GetOdim();
    data_out=last_mach->GetDataOut();
    grad_out=last_mach->GetGradOut();
  }

  activ_forw.pop_back();
  activ_backw.pop_back();

  return del_mach;
}

//-----------------------------------------------
// Insert a machine a an arbitrary position
//-----------------------------------------------

void MachSeq::MachInsert(Mach *new_mach, size_t pos)
{
  if (machs.empty())
    Error("MachSeq::MachInsert() can't insert machine into empty sequence");

  if (pos<1 || pos>=machs.size())
    ErrorN("MachSeq::MachInsert() position must be in [%d,%zu], %zu was requested\n",1,machs.size(),pos);

  Mach *prev_mach=machs[pos-1];
  Mach *next_mach=machs[pos];

  if (prev_mach->GetOdim()!=new_mach->GetIdim()) {
    cout << "Current sequential machine:" << endl; Info(false);
    cout << "Newly added machine:" << endl; new_mach->Info(false);
    Error("input dimension of new sequential machine does not match");
  }
  if (next_mach->GetIdim()!=new_mach->GetOdim()) {
    cout << "Current sequential machine:" << endl; Info(false);
    cout << "Newly added machine:" << endl; new_mach->Info(false);
    Error("output dimension of new sequential machine does not match");
  }
  if (bsize!=new_mach->GetBsize()) {
    cout << "Current sequential machine:" << endl; Info(false);
    cout << "Newly added machine:" << endl; new_mach->Info(false);
    Error("bunch size of new sequential machine does not match");
  }

  machs.insert(machs.begin()+pos,new_mach);

    // connect new machine to the previous one
  new_mach->SetDataIn(prev_mach->GetDataOut());
  prev_mach->SetGradOut(new_mach->GetGradIn());

    // connect new machine to the next one
  next_mach->SetDataIn(new_mach->GetDataOut());
  new_mach->SetGradOut(next_mach->GetGradIn());

  activ_forw.insert(activ_forw.begin()+pos,true);
  activ_backw.insert(activ_backw.begin()+pos,true);
}

//-----------------------------------------------
// File input
//-----------------------------------------------

void MachSeq::ReadData(istream &inpf, size_t s, int bs)
{
  MachMulti::ReadData(inpf, s, bs);

  
  int nbm=machs.size();
  idim = machs[0]->GetIdim();
  bsize = machs[0]->GetBsize();
  odim = machs[nbm-1]->GetOdim();

    // connect first to the outside world
  data_in=machs[0]->GetDataIn();
  grad_in=machs[0]->GetGradIn();
 
    // forward chain the data
  for (int m=1; m<nbm; m++) machs[m]->SetDataIn(machs[m-1]->GetDataOut());
    // backward chain the gradients
  for (int m=nbm-1; m>0; m--) machs[m-1]->SetGradOut(machs[m]->GetGradIn());

    // connect last machine to the outside world
  data_out=machs[nbm-1]->GetDataOut();
  grad_out=machs[nbm-1]->GetGradOut();
}

//
// Tools
//

void MachSeq::Info(bool detailed, char *txt)
{
  if (detailed) {
    cout << "Information on sequential machine" << endl;
    MachMulti::Info(detailed,txt);
  }
  else {
    printf("%sSequential machine [%u] %d- .. -%d, bs=%d, passes=%lu/%lu", txt, (uint) machs.size(), idim, odim, bsize, nb_forw, nb_backw);
    //printf(", this=%p",this);
    tm.disp(", ");
    tbackw.disp(" + back: ");
    printf("\n");
    char ntxt[512];
    sprintf(ntxt,"%s  ", txt);
    for (unsigned int i=0; i<machs.size(); i++) machs[i]->Info(detailed, ntxt);
  }
  printf("%stotal number of parameters: %lu (%d MBytes)\n", txt, GetNbParams(), (int) (GetNbParams()*sizeof(REAL)/1048576));
}

void MachSeq::Forw(int eff_bsize, bool in_train)
{
  if (machs.empty())
    Error("called Forw() for an empty sequential machine");

  tm.start();

  for (unsigned int i=0; i<machs.size(); i++) {
     if (activ_forw[i]) machs[i]->Forw(eff_bsize, in_train);
  }
  nb_forw += (eff_bsize<=0) ? bsize : eff_bsize;

  tm.stop();
}

void MachSeq::Backw(const float lrate, const float wdecay, int eff_bsize)
{
  if (machs.empty())
    Error("called Backw() for an empty sequential machine");

  debugMachOutp("MachSeq Grad",grad_out,idim,odim,eff_bsize);
  tbackw.start();

  for (int i=machs.size()-1; i>=0; i--) {
    if (activ_backw[i]) machs[i]->Backw(lrate,wdecay,eff_bsize);
  }
  nb_backw += (eff_bsize<=0) ? bsize : eff_bsize;

  tbackw.stop();
  debugMachInp("MachSeq Grad",grad_in,idim,odim,eff_bsize);
}
