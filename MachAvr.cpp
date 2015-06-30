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
 */

using namespace std;
#include <iostream>

#include "Tools.h"
#include "MachAvr.h"


void MachAvr::do_alloc()
{
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  if (data_out) cublasFree(data_out);
  if (winner) cublasFree(winner);
  if (grad_in) cublasFree(grad_in);

  data_out = Gpu::Alloc(odim*bsize, "output data of multi-average machine");
  winner = Gpu::Alloc(odim*bsize, "winner of multi-average machine");
  grad_in = Gpu::Alloc(idim*bsize, "input gradient of multi-average machine");

#else
  if (data_out) delete [] data_out;
  if (winner) delete [] winner;
  if (grad_in) delete [] grad_in;
  data_out = (odim*bsize>0) ? new REAL[odim*bsize] : NULL;
  winner = (odim*bsize>0) ? new REAL[odim*bsize] : NULL;
  grad_in = (idim*bsize>0) ? new REAL[idim*bsize] : NULL;
#endif
}


/*
 * constructor
 */

MachAvr::MachAvr()
 : MachCombined()
{
}

/*
 * copy constructor
 * create a copy of the machine without submachines
 */

MachAvr::MachAvr(const MachAvr &m)
 : MachCombined(m)
{
}

/*
 * destructor
 */

MachAvr::~MachAvr()
{
   // data_out and grad_in will be deleted by the desctuctor of Mach
}

/*
 * create a copy of the machine and all submachines
 */

MachAvr *MachAvr::Clone()
{
  MachAvr *m = new MachAvr(*this);
  if (m != NULL)
    m->CloneSubmachs(*this);
  return m;
}

/*
 * set pointer of input data
 * all machines point to the same input
 */

void MachAvr::SetDataIn(REAL *data)
{
  data_in=data;
  for (vector<Mach*>::iterator mit=machs.begin(); mit<machs.end(); mit++)
    (*mit)->SetDataIn(data);
}
 
// set pointer of output gradient
void MachAvr::SetGradOut(REAL *data)
{
  grad_out=data;
  if (machs.size() > 0) machs.back()->SetGradOut(data);
}

/*
 * add a machine to the set
 */

void MachAvr::MachAdd(Mach *new_mach)
{
  if (machs.empty()) {
    machs.push_back(new_mach);
	// think about freeing memory
    idim=new_mach->GetIdim();
    bsize=new_mach->GetBsize();
    data_in=new_mach->GetDataIn();	// TODO
    grad_in=new_mach->GetGradIn();
    do_alloc();
  }
  else {
    if (new_mach->GetIdim() != idim) 
      ErrorN("input dimension of new average machine does not match (%d), should be %d",new_mach->GetIdim(),idim);
    if (new_mach->GetOdim() != idim)
      ErrorN("output dimension of new average machine does not match (%d), should be %d",new_mach->GetOdim(),idim);
    if (bsize!=new_mach->GetBsize()) {
      ErrorN("bunch size of new average machine does not match (%d), should be %d",new_mach->GetBsize(),bsize);
    }
    machs.push_back(new_mach);
 
      // connect TODO
    new_mach->SetDataIn(data_in); // all machines have same input
    new_mach->SetGradOut(NULL); // TODO

     // no new allocation is needed since idim and odim don't change
  }

  activ_forw.push_back(true);
  activ_backw.push_back(true);
}

/*
 * delete last machine from the set
 */

Mach *MachAvr::MachDel()
{
  if (machs.empty()) {
    Error("impossible to delete element from average machine: is already empty");
  }
  
  Mach *del_mach=machs.back();
  machs.pop_back();

  if (machs.empty()) {
    idim=odim=bsize=0;
    data_in=data_out=grad_in=grad_out=NULL;
  }

  activ_forw.pop_back();
  activ_backw.pop_back();

  return del_mach;
}

//-----------------------------------------------
// File input
//-----------------------------------------------

void MachAvr::ReadData(istream &inpf, size_t s, int bs)
{
  MachCombined::ReadData(inpf, s, bs);

  idim = machs[0]->GetIdim();
  bsize = machs[0]->GetBsize();
  odim = machs[0]->GetOdim();
  do_alloc();

    // connect first to the outside world
  MachAvr::SetDataIn(data_in);	// TODO: check
  // TODO: grad_in=machs[0]->GetGradIn();
 
    // connect last machine to the outside world
  //data_out=  TODO
  //grad_out=
}

//
// Tools
//

void MachAvr::Info(bool detailed, char *txt)
{
  if (detailed) {
    cout << "Information on multiple average machine" << endl;
    MachCombined::Info(detailed,txt);
  }
  else {
    printf("%sMultiple average machine [%u] %d- .. -%d, bs=%d, passes=%lu/%lu", txt, (uint) machs.size(), idim, odim, bsize, nb_forw, nb_backw);
    tm.disp(", ");
    tbackw.disp(" + back: ");
    printf("\n");
    char ntxt[512];
    sprintf(ntxt,"%s  ", txt);
    for (unsigned int i=0; i<machs.size(); i++) machs[i]->Info(detailed, ntxt);
  }
  printf("%stotal number of parameters: %lu (%d MBytes)\n", txt, GetNbParams(), (int) (GetNbParams()*sizeof(REAL)/1048576));
}

/*
 * Forward pass
 */

void MachAvr::Forw(int eff_bsize, bool in_train)
{
  if (machs.empty())
    Error("called Forw() for an empty multiple average machine");

  tm.start();
  for (size_t i=0; i<machs.size(); i++) {
     if (!activ_forw[i]) Error("MachAvr::Forw(): deactivation of machines is not supported\n");
     machs[i]->Forw(eff_bsize,in_train);
  }

    // take elementwise max
#ifdef BLAS_CUDA
  //TODO
#else
  vector<REAL*> moptr;	// pointers on the output of the machines
  REAL *optr=data_out;	// create maximized output
  for (size_t i=0; i<machs.size(); i++) moptr.push_back(machs[i]->GetDataOut());

    // TODO: vectorize and consider deactivated machines in an efficient WAY
  for (int b=0; b<odim*eff_bsize; b++) {
    REAL max=moptr[0][b];
    for (size_t i=0; i<machs.size(); i++) {
      if (moptr[i][b]>max) {
        max=moptr[i][b];
        winner[b]=i; // remember index i
      }
    }
    *optr++=max;
  }
#endif
  // TODO nb_forw += (eff_bsize<=0) ? bsize : eff_bsize;
  tm.stop();
}

void MachAvr::Backw(const float lrate, const float wdecay, int eff_bsize)
{
  if (machs.empty())
    Error("called Backw() for an empty average machine");

  debugMachOutp("MachAvr Grad",grad_out,idim,odim,eff_bsize);
  tbackw.start();

  for (int i=machs.size()-1; i>=0; i--) {
    if (activ_backw[i]) machs[i]->Backw(lrate,wdecay,eff_bsize);
  }
  nb_backw += (eff_bsize<=0) ? bsize : eff_bsize;

  tbackw.stop();
  debugMachInp("MachAvr Grad",grad_in,idim,odim,eff_bsize);
}
