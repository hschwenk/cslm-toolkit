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
#include "MachCombined.h"


void MachCombined::do_alloc()
{
  debug2("do_alloc MachCombined %d x %d\n",idim,odim);
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  if (data_out) cublasFree(data_out);
  if (winner) cublasFree(winner);
  if (grad_in) cublasFree(grad_in);

  data_out = Gpu::Alloc(odim*bsize, "output data of a combined machine");
  winner = Gpu::Alloc(odim*bsize, "winner of a combined machine");
  grad_in = Gpu::Alloc(idim*bsize, "input gradient of a combined machine");

  debug2(" - CUDA data_out alloc %lu bytes at %p\n",sizeof(REAL)*odim*bsize,(void*) data_out);
  debug2(" - CUDA grad_in  alloc %lu bytes at %p\n",sizeof(REAL)*idim*bsize,(void*) grad_in);
#else
  if (data_out) delete [] data_out;
  if (winner) delete [] winner;
  if (grad_in) delete [] grad_in;
  data_out = (odim*bsize>0) ? new REAL[odim*bsize] : NULL;
  winner = (odim*bsize>0) ? new REAL[odim*bsize] : NULL;
  grad_in = (idim*bsize>0) ? new REAL[idim*bsize] : NULL;
  debug2(" - data_out alloc %lu bytes at %p\n",sizeof(REAL)*odim*bsize,(void*) data_out);
  debug2(" - grad_in  alloc %lu bytes at %p\n",sizeof(REAL)*idim*bsize,(void*) grad_in);
#endif
}


/*
 * constructor
 */

MachCombined::MachCombined()
 : MachMulti(), winner(NULL)
{
  debug0("*** constructor MachCombined\n");
}

/*
 * copy constructor
 * create a copy of the machine without submachines
 */

MachCombined::MachCombined(const MachCombined &m)
 : MachMulti(m), winner(NULL)
{
  debug0("*** copy constructor MachCombined\n");
}

/*
 * destructor
 */

MachCombined::~MachCombined()
{
  debug1("*** destructor MachCombined %lx\n", (luint) this);
   // data_out and grad_in will be deleted by the desctuctor of Mach
}

/*
 * create a copy of the machine and all submachines
 */

MachCombined *MachCombined::Clone()
{
  MachCombined *m = new MachCombined(*this);
  if (m != NULL)
    m->CloneSubmachs(*this);
  return m;
}

/*
 * set pointer of input data
 * all machines point to the same input
 */

void MachCombined::SetDataIn(REAL *data)
{
  data_in=data;
  for (vector<Mach*>::iterator mit=machs.begin(); mit<machs.end(); mit++)
    (*mit)->SetDataIn(data);
}
 
// set pointer of output gradient
void MachCombined::SetGradOut(REAL *data)
{
  grad_out=data;
  if (machs.size() > 0) machs.back()->SetGradOut(data);
}

/*
 * add a machine to the set
 */

void MachCombined::MachAdd(Mach *new_mach)
{
  if (machs.empty()) {
    debug0("*** add first element to MachCombined\n");
    machs.push_back(new_mach);
	// think about freeing memory
    idim=new_mach->GetIdim();
    bsize=new_mach->GetBsize();
    data_in=new_mach->GetDataIn();	// TODO
    grad_in=new_mach->GetGradIn();
    do_alloc();
  }
  else {
    debug0("*** add new element to MachCombined\n");
    if (new_mach->GetIdim() != idim) 
      ErrorN("input dimension of new combined machine does not match (%d), should be %d",new_mach->GetIdim(),idim);
    if (new_mach->GetOdim() != idim)
      ErrorN("output dimension of new combined machine does not match (%d), should be %d",new_mach->GetOdim(),idim);
    if (bsize!=new_mach->GetBsize()) {
      ErrorN("bunch size of new combined machine does not match (%d), should be %d",new_mach->GetBsize(),bsize);
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

Mach *MachCombined::MachDel()
{
  if (machs.empty()) {
    Error("impossible to delete element from combined machine: is already empty");
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

void MachCombined::ReadData(istream &inpf, size_t s, int bs)
{
  debug0("*** read data of MachCombined\n");
  MachMulti::ReadData(inpf, s, bs);

  idim = machs[0]->GetIdim();
  bsize = machs[0]->GetBsize();
  odim = machs[0]->GetOdim();
  do_alloc();

    // connect first to the outside world
  MachCombined::SetDataIn(data_in);	// TODO: check
  // TODO: grad_in=machs[0]->GetGradIn();
 
    // connect last machine to the outside world
  //data_out=  TODO
  //grad_out=
}

//
// Tools
//

void MachCombined::Info(bool detailed, char *txt)
{
  if (detailed) {
    cout << "Information on multiple combined machine" << endl;
    MachMulti::Info(detailed,txt);
  }
  else {
    printf("%sMultiple combined machine [%u] %d- .. -%d, bs=%d, passes=%lu/%lu", txt, (uint) machs.size(), idim, odim, bsize, nb_forw, nb_backw);
    tm.disp(", ");
    tbackw.disp(" + back: ");
    printf("\n");
    debug5("*** %s   data: %p -> %p, grad %p <- %p\n", txt, (void*)data_in, (void*)data_out, (void*)grad_in, (void*)grad_out);
    char ntxt[512];
    sprintf(ntxt,"%s  ", txt);
    for (unsigned int i=0; i<machs.size(); i++) machs[i]->Info(detailed, ntxt);
  }
  printf("%stotal number of parameters: %lu (%d MBytes)\n", txt, GetNbParams(), (int) (GetNbParams()*sizeof(REAL)/1048576));
}

/*
 * Forward pass
 */

void MachCombined::Forw(int eff_bsize, bool in_train)
{
  debug2("* MachCombined::Forw: %p -> %p\n", (void*) data_in, (void*) data_out);
  if (machs.empty())
    Error("called Forw() for an empty multiple combined machine");

  tm.start();
  for (size_t i=0; i<machs.size(); i++) {
     if (!activ_forw[i]) Error("MachCombined::Forw(): deactivation of combined machines is not supported\n");
     machs[i]->Forw(eff_bsize,in_train);
  }
  nb_forw += (eff_bsize<=0) ? bsize : eff_bsize;

    // we perform no operation to combine the multiple outputs into one
    // THIS MUST BE DONE IN A SPEZIALIZED SUBCLASS
  tm.stop();
}

void MachCombined::Backw(const float lrate, const float wdecay, int eff_bsize)
{
  debug2("* MachCombined::Backw: %p <- %p\n", (void*) grad_in, (void*) grad_out);
  if (machs.empty())
    Error("called Backw() for an empty combined machine");

  debugMachOutp("MachCombined Grad",grad_out,idim,odim,eff_bsize);
  tbackw.start();

  for (int i=machs.size()-1; i>=0; i--) {
     if (!activ_backw[i]) Error("MachCombined::Backw(): deactivation of combined machines is not supported\n");
    if (activ_backw[i]) machs[i]->Backw(lrate,wdecay,eff_bsize);
  }
  nb_backw += (eff_bsize<=0) ? bsize : eff_bsize;

  tbackw.stop();
  debugMachInp("MachCombined Grad",grad_in,idim,odim,eff_bsize);
}
