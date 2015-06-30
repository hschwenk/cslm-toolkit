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
#include "MachMulti.h"

/*******************************************
 *
 ********************************************/

MachMulti::MachMulti()
 : Mach(0,0,0)
{
  machs.clear();
}

MachMulti::MachMulti(const MachMulti &m)
 : Mach(m)
{
  machs.clear();
}

/*******************************************
 *
 ********************************************/

MachMulti *MachMulti::Clone()
{
  MachMulti *m = new MachMulti(*this);
  if (m != NULL)
    m->CloneSubmachs(*this);
  return m;
}

void MachMulti::CloneSubmachs(const MachMulti &mm)
{
  for (unsigned int m=0; m<mm.machs.size(); m++) {
    this->MachAdd( mm.machs[m]->Clone() );
    if (!activ_forw.empty())
      activ_forw.back() = mm.activ_forw[m];
    if (!activ_backw.empty())
      activ_backw.back() = mm.activ_backw[m];
  }
}

/*******************************************
 *
 ********************************************/

MachMulti::~MachMulti()
{
  MachMulti::Delete();
  machs.clear();
}

Mach* MachMulti::MachGet(size_t i)
{
  if (i<0 || i>=machs.size())
    Error("MachMulti: accessing inexistent machine");
  return machs[i];
}

void MachMulti::Delete()
{
  for (unsigned int m=0; m<machs.size(); m++) delete machs[m];
}

void MachMulti::MachAdd(Mach *new_mach)
{
  Error("MachAdd not defined for abstract multiple machine");
}

Mach *MachMulti::MachDel()
{
  Error("MachDel not defined for abstract multiple machine");
  return NULL;
}

ulong MachMulti::GetNbParams() {
  ulong sum=0;

  for (vector<Mach*>::iterator it = machs.begin(); it!=machs.end(); ++it) {
    sum += (*it)->GetNbParams();
  }
  return sum;
}

//-----------------------------------------------
// File output
//-----------------------------------------------


void MachMulti::WriteParams(ostream &of) {
  Mach::WriteParams(of);
  int nbm=machs.size();
  of.write((char*) &nbm, sizeof(int));
}

void MachMulti::WriteData(ostream &outf) {
  int nbm=machs.size(), s=sizeof(REAL);
  outf.write((char*) &nbm, sizeof(int));
  outf.write((char*) &s, sizeof(int));
  for (vector<Mach*>::iterator it = machs.begin(); it!=machs.end(); ++it) {
    (*it)->Write(outf);
  }
}

//-----------------------------------------------
// File input
//-----------------------------------------------

void MachMulti::ReadParams(istream &inpf, bool with_alloc)
{
  if (machs.size() > 0)
    Error("Trying to read multiple machine into non empty data structures\n");

  Mach::ReadParams(inpf, false);
  int nbm;
  inpf.read((char*) &nbm, sizeof(int));
  if (nbm<1) Error("illegal number of machines");
  machs.clear();
  for (int i=0; i<nbm; i++) {
    machs.push_back(NULL);
    activ_forw.push_back(true);
    activ_backw.push_back(true);
  }
}

void MachMulti::ReadData(istream &inpf, size_t s, int bs)
{
  if (s!=machs.size())
    ErrorN("data block of multiple machine has %zu machines (%zu were expected)", s, machs.size());
  
  for (vector<Mach*>::iterator it = machs.begin(); it!=machs.end(); ++it) {
    (*it) = Mach::Read(inpf, bs);
  }
}

//
// Tools
//

void MachMulti::SetBsize(int bs)
{
  if (bs<1) Error("wrong value in SetBsize()");
  Mach::SetBsize(bs);
  for (uint i=0; i<machs.size(); i++) machs[i]->SetBsize(bs);
}

void MachMulti::SetNbEx(ulong nf, ulong nb)
{
  Mach::SetNbEx(nf, nb);
  for (uint i=0; i<machs.size(); i++) machs[i]->SetNbEx(nf, nb);
}

void MachMulti::Info(bool detailed, char *txt)
{
  if (detailed) {
    if (machs.size()) {
      Mach::Info();
      for (unsigned int i=0; i<machs.size(); i++) {
        cout << "MACHINE " << i << ": " << endl;
        machs[i]->Info();
      }
    }
    else
      cout << " *** empty ***" << endl;
  }
  else {
    printf("%sMultiple machine %d- .. -%d, bs=%d, passes=%lu/%lu", txt, idim, odim, bsize, nb_forw, nb_backw);
    tm.disp(", ");
    printf("\n");
    char ntxt[256];
    sprintf(ntxt,"%s  ", txt);
    for (unsigned int i=0; i<machs.size(); i++) machs[i]->Info(detailed, ntxt);
  }
  printf("%stotal number of parameters: %lu (%d MBytes)\n", txt, GetNbParams(), (int) (GetNbParams()*sizeof(REAL)/1048576));
}

bool MachMulti::CopyParams(Mach* mach)
{
  MachMulti* machmulti = static_cast<MachMulti*>(mach);
  size_t nb_machs = this->machs.size();
  if (    Mach::CopyParams(mach)
      && (machmulti->machs.size() == nb_machs) ) {
    this->activ_forw  = machmulti->activ_forw;
    this->activ_backw = machmulti->activ_backw;
    for (size_t i = 0 ; i < nb_machs ; i++)
      if (!(this->machs[i]->CopyParams(machmulti->machs[i])))
        return false;
    return true;
  }
  else
    return false;
}

void MachMulti::Forw(int eff_bsize, bool in_train)
{
  if (machs.empty())
    Error("called Forw() for an empty multiple machine");
  else
    Error("call to Forw() not defined for an abstract multiple machine");
}

void MachMulti::Backw(const float lrate, const float wdecay, int eff_bsize)
{
  Error("call to Backw() not defined for an abstract multiple machine");
}


void MachMulti::Activate(int nb, bool do_forw, bool do_backw)
{
  if (nb<0 || nb>=(int)machs.size())
    Error("MachMulti::Activate: wrong machine number\n");

  activ_forw[nb] = do_forw;
  activ_backw[nb] = do_backw;
}
