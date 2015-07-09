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

using namespace std;
#include <iostream>
#include <sstream>
#include <map>

#include "Tools.h"
#include "MachTab.h"
#include "MachPar.h"

void MachPar::do_alloc()
{
  debug2("do_alloc MachPar %d x %d\n",idim,odim);
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  if (data_out) cublasFree(data_out);
  if (grad_in) cublasFree(grad_in);

  data_out = Gpu::Alloc(odim*bsize, "output data of parallel machine");
  grad_in = Gpu::Alloc(idim*bsize, "input gradient of parallel machine");

  debug2(" - CUDA data_out alloc %lu bytes at %p\n",sizeof(REAL)*odim*bsize,(void*) data_out);
  debug2(" - CUDA grad_in  alloc %lu bytes at %p\n",sizeof(REAL)*idim*bsize,(void*) grad_in);
#else
  if (data_out) delete [] data_out;
  if (grad_in) delete [] grad_in;
  data_out = (odim*bsize>0) ? new REAL[odim*bsize] : NULL;
  grad_in = (idim*bsize>0) ? new REAL[idim*bsize] : NULL;
  debug2(" - data_out alloc %lu bytes at %p\n",sizeof(REAL)*odim*bsize,(void*) data_out);
  debug2(" - grad_in  alloc %lu bytes at %p\n",sizeof(REAL)*idim*bsize,(void*) grad_in);
#endif
}


MachPar::MachPar()
 : MachMulti()
{
  debug0("** constructor MachPar\n");
}

MachPar::MachPar(const MachPar &m)
 : MachMulti(m)
{
  debug0("** copy constructor MachPar\n");
}

MachPar::~MachPar()
{
  debug0("** destructor MachPar\n");
  // data_out and grad_in will be freed by Mach::~Mach()
  for (unsigned int m=0; m<machs.size(); m++)
  {
#ifdef BLAS_CUDA
    if (machs[m]->GetDataIn ()) cublasFree(machs[m]->GetDataIn ());
    if (machs[m]->GetGradOut()) cublasFree(machs[m]->GetGradOut());
#else
    if (machs[m]->GetDataIn ()) delete [] machs[m]->GetDataIn ();
    if (machs[m]->GetGradOut()) delete [] machs[m]->GetGradOut();
#endif
  }
}

MachPar *MachPar::Clone()
{
  MachPar *m = new MachPar(*this);
  if (m != NULL)
    m->CloneSubmachs(*this);
  return m;
}
 
void MachPar::MachAdd(Mach *new_mach)
{
  if (machs.empty()) {
    debug0("** add first element to parallel machine\n");
    machs.push_back(new_mach);
	// think about freeing memory
    idim=new_mach->GetIdim();
    odim=new_mach->GetOdim();
    bsize=new_mach->GetBsize();
    data_in=NULL; // will be set by MachPar::SetDataIn()
    data_out=NULL;
    grad_in = NULL;
    grad_out = NULL;
    do_alloc();
  }
  else {
    debug0("** add new element to parallel machine\n");
    if (bsize!=new_mach->GetBsize())
      Error("bunch size of new parallel machine does not match");
    machs.push_back(new_mach);
 
      // resize input gradient and output data
    idim += new_mach->GetIdim();
    odim += new_mach->GetOdim();
    do_alloc();
  }
#ifdef BLAS_CUDA
  new_mach->SetDataIn (Gpu::Alloc(bsize * new_mach->GetIdim() * sizeof(REAL), "submachine input"));
  new_mach->SetGradOut(Gpu::Alloc(bsize * new_mach->GetOdim() * sizeof(REAL), "submachine output gradients"));
#else
  new_mach->SetDataIn (new REAL[bsize * new_mach->GetIdim()]);
  new_mach->SetGradOut(new REAL[bsize * new_mach->GetOdim()]);
#endif
  activ_forw.push_back(true);
  activ_backw.push_back(true);
}

Mach *MachPar::MachDel()
{
  if (machs.empty()) {
    Error("impossible to delete element from parallel machine: is already empty");
  }
  
  Error("TODO");
  activ_forw.pop_back();
  activ_backw.pop_back();
  return NULL;
}


//-----------------------------------------------
// File output
//-----------------------------------------------

void MachPar::ReadData(istream &inpf, size_t s, int bs)
{
  debug0("* read data of MachPar\n");
  MachMulti::ReadData(inpf, s, bs);

     // calculate idim and odim and allocate data_out and grad_in
  idim=odim=0;
  for (uint m=0; m<machs.size(); m++) {
    idim += machs[m]->GetIdim();
    odim += machs[m]->GetOdim();
  }
  bsize = machs[0]->GetBsize();
  do_alloc();
  for (unsigned int m=0; m<machs.size(); m++) {
#ifdef BLAS_CUDA
    machs[m]->SetDataIn (Gpu::Alloc(bsize * machs[m]->GetIdim() * sizeof(REAL), "submachine input"));
    machs[m]->SetGradOut(Gpu::Alloc(bsize * machs[m]->GetOdim() * sizeof(REAL), "submachine output gradients"));
#else
    machs[m]->SetDataIn (new REAL[bsize * machs[m]->GetIdim()]);
    machs[m]->SetGradOut(new REAL[bsize * machs[m]->GetOdim()]);
#endif
  }

  // this is no more needed -> everything is done in Mach 
/*
  // scanning for MachTab with shared addresses
  std::map<int, REAL*> tadr;
  for (uint m=0; m<machs.size(); m++) {
    MachTab *mt= (MachTab*) machs[m];
    if (mt->GetMType()==file_header_mtype_tab) {
      if(Mach::fileid >= file_header_version3){
	if (tadr[mt->GetShareId()] == NULL) {
	  debug3("Storing address (%p) of machine %d with share-id %d\n",mt->GetTabAdr(),m, mt->GetShareId());
	  tadr[mt->GetShareId()] = mt->GetTabAdr();
	    if(mt->GetTabAdr() == NULL) {
	      std::stringstream oss ("In MachPar: machine "); 
	      oss << m << " should have its weights allocated!\n";
	      Error(oss.str().c_str());
	    }
	} else { 
	  debug3("Setting address (%p) of machine %d with share-id %d\n",mt->GetTabAdr(),m, mt->GetShareId());
	  mt->SetTabAdr(tadr[mt->GetShareId()]);
        } 
	*//*else {
	    debug3("Machine %d with share-id '%s' already has its own weights at address (%p)\n",m, mt->GetShareId(), mt->GetTabAdr());
	    if(mt->GetTabAdr() == NULL) {
	      //std::ostringstream oss("In MachPar: machine ");
	      std::stringstream oss ("In MachPar: machine "); 
	      oss << m << " should have its weights allocated!\n";
	      Error(oss.str().c_str());
	    }
	  }*/
        /*} else { // before file_header_version3, all MachTab in a MachPar share the weights
	    if(tadr[-1] == NULL ){
		if(tadr[-1]) { debug2("Storing further address (%p) of machine %d\n",tadr[-1],m); } //  cout << "set NEW tadr" << endl; }
		else { debug2("Storing address (%p) of machine %d\n",mt->GetTabAdr(),m); } //cout << "set tadr" << endl; }
		tadr[-1]=mt->GetTabAdr();
	    } else {
		debug2("setting address of machine %d to %p\n",m,tadr[-1]);
		//cout << "set address of machine " << m << " to " << tadr[-1] << endl;
		//mt->FreeTabAdr();
		mt->SetTabAdr(tadr[-1]);
	    }
	  }
      } //if file_header_mtype_tab 
    } // for all machines 
*/
}

//
// Tools
//

void MachPar::Info(bool detailed, char *txt)
{
  if (detailed) {
    cout << "Information on parallel machine" << endl;
    MachMulti::Info(detailed);
  }
  else {
    printf("%sParallel machine %d-%d, bs=%d, passes=%lu/%lu", txt, idim, odim, bsize, nb_forw, nb_backw);
    tm.disp(", ");
    printf("\n");
    debug5("%s   data: %p -> %p, grad %p <- %p\n", txt, (void*)data_in, (void*)data_out, (void*)grad_in, (void*)grad_out);
    char ntxt[512];
    sprintf(ntxt,"%s  ", txt);
    for (unsigned int i=0; i<machs.size(); i++) machs[i]->Info(detailed, ntxt);
  }
  printf("%stotal number of parameters: %lu (%d MBytes)\n", txt, GetNbParams(), (int) (GetNbParams()*sizeof(REAL)/1048576));
}

// TODO we do not organize correcty the input in the forward and backward pass with bunch mode !
// TODO since this is wrongly done at the input and output we finally get the corrrect result
// TODO but only when combine identical machines (like MachTab with shared codes)


// forward pass for all machines and copy output into cumulated output
void MachPar::Forw(int eff_bsize, bool in_train)
{
  debug4("** MachPar::Forw: %p[%d] -> %p[%d]\n",(void*)data_in,idim,(void*)data_out,odim);
  if (machs.empty())
    Error("called Forw() for an empty parallel machine");

  debugMachInp("MachPar",data_in,idim,odim,eff_bsize);

  tm.start();
  if (eff_bsize<=0) eff_bsize=bsize;

#ifdef BLAS_CUDA
  Gpu::StreamSynchronize();
#endif

      // copy the input data of MachPar to the individual machines
      // (they have their own input to ensure correct organisations of the batches)
  REAL *iptr=data_in;
  for (unsigned int m=0; m<machs.size(); m++) {
    int m_idim = machs[m]->GetIdim();
    if (activ_forw[m]) {
      REAL* Mach_Data_In_Ptr= machs[m]->GetDataIn();
#ifdef BLAS_CUDA
      Gpu::SetConfig(machs[m]->GetGpuConfig());
      Gpu::Memcpy2DAsync(Mach_Data_In_Ptr, m_idim*sizeof(REAL),
                 iptr, idim*sizeof(REAL),
                 m_idim*sizeof(REAL), eff_bsize,
                 cudaMemcpyDeviceToDevice);
      Gpu::CheckError("MachPar::Forw - After copying input of sub-machine");
#else
      for(int i=0; i<eff_bsize; i++)
        memcpy(Mach_Data_In_Ptr + i*m_idim, iptr + i*idim, m_idim*sizeof(REAL));
#endif
    }
    iptr += m_idim;
  }

    // forward all machines
  for (unsigned int m=0; m<machs.size(); m++) {
    if (activ_forw[m]) {
      debug1("  MachPar[%d]: forward mach\n",m);
      machs[m]->Forw(eff_bsize,in_train);
    }
    else {
        // set output of inactive machines to zero
#ifdef BLAS_CUDA
      Gpu::MemSet(machs[m]->GetDataOut(), 0, machs[m]->GetOdim()*eff_bsize);
#else
      memset(machs[m]->GetDataOut(), 0, machs[m]->GetOdim()*eff_bsize);
#endif
    }
  }

      // copy the output data of the individual machines to MachPar's output
      // we also do this for inactive machines to preserve the zero output
  REAL *optr=data_out;
  for (unsigned int m=0; m<machs.size(); m++) {
    int m_odim = machs[m]->GetOdim();
#ifdef BLAS_CUDA
    Gpu::SetConfig(machs[m]->GetGpuConfig());
    Gpu::Memcpy2DAsync(optr, odim*sizeof(REAL), machs[m]->GetDataOut(), m_odim*sizeof(REAL), m_odim*sizeof(REAL), eff_bsize, cudaMemcpyDeviceToDevice);
    Gpu::CheckError("MachPar::Forw - After copying output of sub-machine");
#else
    for (int i=0; i<eff_bsize; i++)
      memcpy(optr+i*odim, machs[m]->GetDataOut()+i*m_odim, m_odim*sizeof(REAL));
#endif
    optr += m_odim;
  }

  nb_forw += eff_bsize; 
  debug0("MachPar::Forw: done\n");

  tm.stop();
  debugMachOutp("MachPar",data_out,idim,odim,eff_bsize);
}

// backward pass for all machines and copy input gradient into cumulated gradient
void MachPar::Backw(const float lrate, const float wdecay, int eff_bsize)
{
  debug4("** MachPar::Backw: %p[%d] <- %p[%d]\n",(void*)grad_in,idim,(void*)grad_out,odim);
  if (machs.empty())
    Error("called Backw() for an empty parallel machine");
  if (eff_bsize<=0) eff_bsize=bsize;
 
  tm.start();

      // copy the output gradients data of MachPar to the individual machines
      // (they have their own gradients to ensure correct organisations of the batches)

  REAL *gptr=grad_out;
  for (unsigned int m=0; m<machs.size(); m++) {
    int m_odim = machs[m]->GetOdim();
    if (activ_backw[m]) {
      REAL* Mach_Grad_Out_Ptr= machs[m]->GetGradOut();
#ifdef BLAS_CUDA
      Gpu::SetConfig(machs[m]->GetGpuConfig());
      Gpu::Memcpy2DAsync(Mach_Grad_Out_Ptr, m_odim*sizeof(REAL),
                 gptr, odim*sizeof(REAL),
                 m_odim*sizeof(REAL), eff_bsize,
                 cudaMemcpyDeviceToDevice);
      Gpu::CheckError("MachPar::Forw - After copying gradient of sub-machine");
#else
      for(int i=0; i<eff_bsize; i++)
        memcpy(Mach_Grad_Out_Ptr + i*m_odim, gptr + i*odim, m_odim*sizeof(REAL));
#endif
    }
    gptr += m_odim;
  }

  for (unsigned int m=0; m<machs.size(); m++) {
    if (activ_backw[m])
      machs[m]->Backw(lrate,wdecay,eff_bsize);
    else {
        // set input gradients of inactive machines to zero
#ifdef BLAS_CUDA
      Gpu::MemSet(machs[m]->GetGradIn(), 0, machs[m]->GetIdim()*eff_bsize);
#else
      memset(machs[m]->GetGradIn(), 0, machs[m]->GetIdim()*eff_bsize);
#endif
    }
  }

      // copy the input gradients of the individual machines to MachPar
  gptr=grad_in;
  for (unsigned int m=0; m<machs.size(); m++) {
    int m_idim = machs[m]->GetIdim();
#ifdef BLAS_CUDA
    Gpu::SetConfig(machs[m]->GetGpuConfig());
    Gpu::Memcpy2DAsync(gptr, idim*sizeof(REAL), machs[m]->GetGradIn(), m_idim*sizeof(REAL), m_idim*sizeof(REAL), eff_bsize, cudaMemcpyDeviceToDevice);
    Gpu::CheckError("MachPar::Forw - After copying output of sub-machine");
#else
    for (int i=0; i<eff_bsize; i++)
      memcpy(gptr+i*idim, machs[m]->GetGradIn()+i*m_idim, m_idim*sizeof(REAL));
#endif
  }

  nb_backw += eff_bsize; 

  tm.stop();
}

