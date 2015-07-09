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
#include <signal.h>
#include <unistd.h>

#include "Tools.h"
#include "Mach.h"
#include "MachCopy.h"
#include "MachTab.h"
#include "MachLin.h"
#include "MachSig.h"
#include "MachTanh.h"
#include "MachSoftmax.h"
#include "MachSoftmaxStable.h"
#include "MachSoftmaxClass.h"
#include "MachLinRectif.h"
#include "MachSeq.h"
#include "MachPar.h"
#include "MachSplit.h"
#include "MachSplit1.h"
#include "MachJoin.h"

vector<Mach*> signal_mach;
int Mach::fileid=-1;
std::map<int, Mach *> prSharedMachines; // to store Mach pointers for sharing using clone() function

#ifdef BLAS_CUDA
# include "Blas.h"
#else

int inc1=1;
#endif

void HandlerSigUSR1(int s) {
  time_t now;
  time(&now); // TODO: ctime is not rentrant ! use ctime_r() instead if needed
  cout << " - catched signal USR1 at " << ctime(&now) << endl;
  signal_mach[0]->Info(false, (char*)" -   ");
  cout.flush();
  //for (uint i=0; i<1; i++) signal_mach[i]->Info(false, (char*)" -   ");
  signal(SIGUSR1, HandlerSigUSR1);
}

//***********************************************

#ifdef BLAS_CUDA
void Mach::do_alloc()
{  
  Gpu::Init();

  debug3("*** do_alloc CUDA Mach type %d: %dx%d\n",GetMType(),idim,odim);
  data_out = Gpu::Alloc(odim*bsize, "output data for a machine");
  debug1("*** - data_out=%p\n",(void*)data_out);
  data_in=NULL; //  should be set later by SetDataIn()
  drop_out_rand = NULL; // will be allocated when calling SetDropOut()
  grad_in = Gpu::Alloc(idim*bsize, "input gradient for a machine");
  debug1("*** - grad_in=%p\n",(void*)grad_in);
  grad_out=NULL; // should be set later by SetGradOut()
}

void Mach::SetDropOut(const REAL v) {
  if (v<0 || v>=1.0) Error("SetDropOut: the value must be in [0,1)");
  if (drop_out_rand) cublasFree(drop_out_rand);
  if (v>0) {
    drop_out_rand = Gpu::Alloc(odim*bsize, "buffer for random values for drop-out");
  }
  drop_out=v;
  debug4("drop_out: %f in %p for %dx%d\n",drop_out,drop_out_rand,idim,odim);
}
#endif

//***********************************************

#ifndef BLAS_CUDA
void Mach::do_alloc()
{
  debug3("*** do_alloc Mach type %d: %dx%d\n",GetMType(),idim,odim);
  if (odim*bsize>0) {
    data_out=::new REAL[odim*bsize];
    if (!data_out) Error ("can't allocate memory for data_out");
    drop_out_rand = NULL; // will be allocated when calling SetDropOut()
  }
  else { data_out=drop_out_rand=NULL; }
  debug1("*** - data_out=%p\n",(void*)data_out);
  data_in=NULL; // should be set later by SetDataIn() 
  if (idim*bsize>0) {
    grad_in=::new REAL[idim*bsize];
    if (!grad_in) Error ("can't allocate memory for grad_in");
  }
  else grad_in=NULL;
  debug1("*** - grad_in=%p\n",(void*)grad_in);
  grad_out=NULL; // (luint) this) should be set later by SetGradOut()
}

void Mach::SetDropOut(const REAL v) {
  if (v<0 || v>=1.0) Error("SetDropOut: the value must be in [0,1)");
  if (drop_out_rand) delete drop_out_rand;
  if (v>0) {
    drop_out_rand = ::new REAL[odim*bsize];
    if (!drop_out_rand) Error ("can't allocate memory for drop_out");
  }
  drop_out=v;
  debug4("drop_out: %f in %p for %dx%d\n",drop_out,drop_out_rand,idim,odim);
}
#endif


Mach::Mach(const int p_idim, const int p_odim, const int p_bsize, const ulong p_nbfw, const ulong p_nbbw)
 : idim(p_idim), odim(p_odim), bsize(p_bsize), nb_forw(p_nbfw), nb_backw(p_nbbw), update(true), lr_coeff(1.0), drop_out(0.0), drop_out_rand(NULL)
{
  debug0("*** constructor Mach\n");
  do_alloc();
#ifdef BLAS_CUDA
  gpu_conf = Gpu::GetConfig();
#endif

    // setup SIGUSR1 handler
  //cout << " - setting up handler for signal USR1" << endl;
  if (signal_mach.empty()) signal(SIGUSR1, HandlerSigUSR1);
  signal_mach.push_back(this);
}

Mach::Mach(const Mach &m, const int p_idim)
{
  debug0("*** copy constructor Mach\n");
  if (p_idim > 0)
    idim = p_idim;
  else
    idim = m.idim;
  odim = m.odim;
  bsize = m.bsize;
  nb_forw = m.nb_forw;
  nb_backw = m.nb_backw;
  update = m.update;
  lr_coeff = m.lr_coeff;
  drop_out = m.drop_out;
  drop_out_rand = NULL;
#ifdef BLAS_CUDA
  gpu_conf = m.gpu_conf; // this is very important ! we share the weights so they must be on the same machine
  Gpu::SetConfig(gpu_conf);
#endif
  do_alloc();
  data_in = m.data_in;
  grad_out = m.grad_out;

    // setup SIGUSR1 handler
  //cout << " - setting up handler for signal USR1" << endl;
  if (signal_mach.empty()) signal(SIGUSR1, HandlerSigUSR1);
  signal_mach.push_back(this);
}

/*******************************************
 *
 ********************************************/

Mach::~Mach()
{
  debug1("*** destructor Mach %lx\n", (luint) this);
#ifdef BLAS_CUDA
  if (data_out) cublasFree(data_out);
  if (drop_out_rand) cublasFree(drop_out_rand);
  if (grad_in) cublasFree(grad_in);
#else
  if (data_out) delete [] data_out;
  if (drop_out_rand) delete [] drop_out_rand;
  if (grad_in) delete [] grad_in;
#endif
  signal_mach.pop_back();	 //TODO: we should search for correct machine and delete it
}

//-----------------------------------------------
// File output
//-----------------------------------------------

void Mach::WriteParams(ostream &of) {
  debug0("*** write params of Mach\n");
    // write machine specific params
  of.write((char*) &nb_forw, sizeof(ulong));
  of.write((char*) &nb_backw, sizeof(ulong));
}

void Mach::WriteData(ostream &of) {
  debug0("*** writing data of general machine to file\n");
  const int i=0, s=sizeof(REAL);
  of.write((char*) &i, sizeof(int));
  of.write((char*) &s, sizeof(int));
}

void Mach::Write(ostream &of)
{
  debug0("*** writing data of general machine to file\n");
  char header[file_header_size];
  for (int i=0; i<file_header_size; i++) header[i]=' ';
  sprintf(header,"%s %d",file_header_name, file_header_version);
  of.write(header,file_header_size);
  of.write((char*) &idim, sizeof(int));
  of.write((char*) &odim, sizeof(int));
  of.write((char*) &bsize, sizeof(int));
  int mtype=GetMType();
  of.write((char*) &mtype, sizeof(int));
  WriteParams(of);
  WriteData(of);
}

//-----------------------------------------------
// File input
//-----------------------------------------------


void Mach::ReadParams(istream &inpf, bool with_alloc)
{
  debug0("*** read params of type Mach\n");
  switch (Mach::fileid) {
    case file_header_version1: // read int but store ulong
      unsigned int itmp;
      inpf.read((char*) &itmp, sizeof(int)); nb_forw = (ulong) itmp;
      inpf.read((char*) &itmp, sizeof(int)); nb_backw = (ulong) itmp;
      debug2("V1 read int counters %lu/%lu\n",nb_forw,nb_backw);
      break;
    case file_header_version2: 
    case file_header_version3: 
    case file_header_version4: 
      inpf.read((char*) &nb_forw, sizeof(ulong));
      inpf.read((char*) &nb_backw, sizeof(ulong));
      debug2("V2 to V4 read ulong counters %lu/%lu\n",nb_forw,nb_backw);
      break;
    default:
      Error("internal error, fileid is unset");
  }
}

void Mach::ReadData(istream &inpf, size_t s, int bs)
{
  // there is nothing to read
}

Mach *Mach::Read(istream &inpf, int bs)
{
  debug0("\n*** reading generic machine from file\n");
  char header[file_header_size], h[file_header_size];
  int v;

  inpf.read(header,file_header_size);
  if (sscanf(header,"%s %d",h,&v) != 2) {
    ErrorN("format of machine file not recognised: %s", header);
  }

  if (Mach::fileid<0) {
      Mach::fileid=v;
  }
  else {
    if (v!=Mach::fileid) ErrorN("all network files must have the same file ID %d",Mach::fileid);
  }
  if (strcmp(h,file_header_name)) {
    ErrorN("unsupported file type (%s), expected '%s'\n", h, file_header_name);
  }
  switch (Mach::fileid) {
    case file_header_version1: 
    case file_header_version2:
    case file_header_version3:
    case file_header_version4:
	break;
    default:
      ErrorN("unsupported version of machine file (%d)\n",Mach::fileid);
  }

    // read idim, odim, bsize 
  int f_idim, f_odim, f_bsize;
  inpf.read((char*) &f_idim, sizeof(int));
  inpf.read((char*) &f_odim, sizeof(int));
  inpf.read((char*) &f_bsize, sizeof(int));
  debug3("*** file read: dim=%d x %d, bs=%d\n",f_idim,f_odim,f_bsize);
  if (bs <= 0)
    bs = f_bsize;

   // read and parse machine type
  int mtype;
  Mach *m=NULL;
  inpf.read((char*) &mtype, sizeof(int));
  switch (mtype) {
    case file_header_mtype_base: m = new Mach(f_idim,f_odim,bs); break;
    case file_header_mtype_copy: m = new MachCopy(f_idim,f_odim,bs); break;
    case file_header_mtype_tab: m = new MachTab(f_idim,f_odim,bs,0,0); break;
    case file_header_mtype_lin: m = new MachLin(f_idim,f_odim,bs); break;
    case file_header_mtype_sig: m = new MachSig(f_idim,f_odim,bs); break;
    case file_header_mtype_tanh: m = new MachTanh(f_idim,f_odim,bs); break;
    case file_header_mtype_softmax: m = new MachSoftmax(f_idim,f_odim,bs); break;
    case file_header_mtype_softmax_stable: m = new MachSoftmaxStable(f_idim,f_odim,bs); break;
    case file_header_mtype_lin_rectif: m = new MachLinRectif(f_idim,f_odim,bs); break;
    case file_header_mtype_softmax_class: m = new MachSoftmaxClass(f_idim, f_odim, bs); break;
    case file_header_mtype_multi: m = new MachMulti(); break;
    case file_header_mtype_mseq: m = new MachSeq(); break;
    //case file_header_mtype_mstack: m = new MachStack; break;
    case file_header_mtype_mpar: m = new MachPar(); break;
    case file_header_mtype_msplit1: m = new MachSplit1; break;
    case file_header_mtype_msplit: m = new MachSplit; break;
    case file_header_mtype_mjoin: m = new MachJoin; break;
    default:
      ErrorN("unknown machine type in file (%d)", mtype);
  }
  if (!m) Error("no valid machine loaded");

    // read rest of (machine specific) params
  m->ReadParams(inpf);

  int s;
  inpf.read((char*) &s,sizeof(int));  // number of elements
  inpf.read((char*) &v,sizeof(int));  // size in bytes of each element
  if (v != sizeof(REAL)) {
    ErrorN( "binary data on file uses %d bytes while the current code is compiled for %lu bytes\n", v, sizeof(REAL));
  }
 
    //Loic: handling special case of MachTab
    if(m->GetMType() == file_header_mtype_tab){
	MachTab* mt = static_cast<MachTab*>(m);
	// if version > 3 then check share-id
	if(Mach::fileid >= file_header_version3){
	    m->ReadData(inpf, s, bs);
	    if(prSharedMachines[mt->GetShareId()] == NULL){
		//fprintf(stderr, " ... new primary MachTab with share-id %d\n", mt->GetShareId());
		prSharedMachines[mt->GetShareId()] = mt;
		if(mt->GetTabAdr() == NULL) {
		    Error("Mach::Read: machine should have its weights allocated!\n");
		}
	    } else {
		//fprintf(stderr, " ... cloning secondary MachTab with share-id %d\n", mt->GetShareId());
		m = prSharedMachines[mt->GetShareId()]->Clone();
	    }
	
        } else { // before file_header_version3, all MachTab in a MachPar share the weights
	    
	    if(prSharedMachines[-1] == NULL ){
		if(mt->bExternal==0)  m->ReadData(inpf, s, bs); //read the data for the first MachTab
		else{
		    Error("The first MachTab should have its own data but is set to have external data\n");
		}
		debug2("Storing address (%p) of machine %d\n",mt->GetTabAdr(),m); 
		prSharedMachines[-1]=m;
	    } else {
		m = prSharedMachines[-1]->Clone();
		debug1(" cloning MachTab, address =  %p\n", mt->GetTabAdr());
		//fprintf(stderr, " cloning MachTab, address =  %p\n", mt->GetTabAdr());
	    }
	  }
    }
    else if(Mach::fileid >= file_header_version4 && Mach::canShare(mtype)) { 
	//fprintf(stderr, "Shareable machine mtype = %d\n", mtype);
	Shareable* sharem = dynamic_cast<Shareable*>(m);
	//fprintf(stderr, "Shareable: external=%d  share-id=%d\n", sharem->HasExternalData(), sharem->GetShareId());
	if(sharem->HasExternalData()){
	    if(prSharedMachines[sharem->GetShareId()] != NULL){
		//fprintf(stderr, " ... secondary machine with share-id %d -> cloning primary machine\n", sharem->GetShareId());
		m = (MachLin*)prSharedMachines[sharem->GetShareId()]->Clone();
	    } else {
		ErrorN("Found a secondary machine with shareid=%d, but the primary machine is not yet created\n", sharem->GetShareId());
	    }
	} else { 
	    if(sharem->GetShareId() != -1){
		//fprintf(stderr, " ... new primary machine with share-id %d\n", sharem->GetShareId());
		prSharedMachines[sharem->GetShareId()] = m;
	    } 
	    //else { fprintf(stderr, " ... new primary machine with no sharing\n"); }
	    m->ReadData(inpf, s, bs);
	}
    } else { 
	//fprintf(stderr, " ... new machine without sharing type=%d\n", m->GetMType());
	m->ReadData(inpf, s, bs);
	// TODO: check EOF
    }
    return m;
}

//-----------------------------------------------
// Tools
//-----------------------------------------------

void Mach::Info(bool detailed, char *txt)
{
  if (detailed) {
    cout << " - dimensions: in=" << idim << ", out=" << odim << endl;
    cout << " - number of parallel examples=" << bsize << endl;
    if (drop_out>0)
      cout << " - drop-out: " <<  drop_out << endl;
    cout << " - number of passes: " << nb_forw << "/" << nb_backw << endl;
  }
  else {
    if (drop_out>0)
      printf("%sMach %d-%d, bs=%d, drop-out=%4.2f, passes=%lu/%lu", txt, idim, odim, bsize, drop_out, nb_forw, nb_backw);
    else
      printf("%sMach %d-%d, bs=%d, passes=%lu/%lu", txt, idim, odim, bsize, nb_forw, nb_backw);
    if (lr_coeff != 1.0) printf(", lrate-coeff=%.2f", lr_coeff);
#ifdef BLAS_CUDA
    printf(", on GPU %d", Gpu::GetCudaDevice(Gpu::GetDevice(gpu_conf)));
#endif
    tm.disp(", ");
    printf("\n");
    debug5("*** %s   data: %p -> %p, grad %p <- %p\n", txt, (void*)data_in, (void*)data_out, (void*)grad_in, (void*)grad_out);
  }
}

bool Mach::CopyParams(Mach* mach)
{
  // type, idim, odim and bsize must be equals
  if (   (NULL != mach)
      && (mach->GetMType() == this->GetMType())
      && (mach->idim  == this->idim )
      && (mach->odim  == this->odim )
      && (mach->bsize == this->bsize) ) {
    this->nb_forw  = mach->nb_forw;
    this->nb_backw = mach->nb_backw;
    this->update   = mach->update;
    return true;
  }
  else
    return false;
}

//-----------------------------------------------
// Training
//-----------------------------------------------

void Mach::Forw(int eff_bsize, bool in_train)
{
  if (idim!=odim)
    Error("Mach::Forw(): call to default Forw() function with different dimensions");
  if (eff_bsize<=0) eff_bsize=bsize;
  if (!data_in)
    Error("Mach::Forw(): input data is not set");

  tm.start();

#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  COPY(eff_bsize*idim,data_in,1,data_out,1); // this does work on host or GPU
#else
  int dim=eff_bsize*idim;
  COPY(&dim,data_in,&inc1,data_out,&inc1); // this does work on host or GPU
#endif
  nb_forw += (ulong) eff_bsize;

  tm.stop();
}

void Mach::Backw (const float lrate, const float wdecay, int eff_bsize)
{
  if (idim!=odim)
    Error("Mach::Backw(): call to default Train() function with different dimensions");
  if (!grad_out)
    Error("Mach::Backw(): output gradient is not set");

  if (eff_bsize<=0) eff_bsize=bsize;
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  COPY(eff_bsize*idim,grad_out,1,grad_in,1);
#else
  memcpy(grad_in,grad_out,eff_bsize*idim*sizeof(REAL));
#endif
  nb_backw += (ulong) eff_bsize;
}

//******************************************

void GpuUnlock()
{
#ifdef BLAS_CUDA
  Gpu::Unlock();
#endif
}

//***********************************************
// Find sub-machines matching desired mtype in parent_mach (depth-first).

// Returns the first sub-machine found (depth-first).
// Returns NULL if none is found.
Mach* FindFirstMatching(int mtype, Mach* parent_mach)
{
  MachMulti* mach_multi = NULL;
  if (parent_mach->GetMType() == mtype) {
    return parent_mach;
  }
  else if ((mach_multi = dynamic_cast<MachMulti*>(parent_mach))) {
    // Maybe a sub-machine will have the right mtype
    int nb_sub_mach = mach_multi->MachGetNb();
    for (int i=0; i<nb_sub_mach; i++) {
      Mach* found_mach = FindFirstMatching(mtype, mach_multi->MachGet(i));
      if (found_mach != NULL) {
        return found_mach;
      }
    }
  }
  return NULL;
}

// Helper function for FindAllMatching
void EnqueueAllMatching(int mtype, Mach* parent_mach, std::vector<Mach*> queue)
{
  MachMulti* mach_multi = NULL;
  if (parent_mach->GetMType() == mtype) {
    queue.push_back(parent_mach);
  }
  if ((mach_multi = dynamic_cast<MachMulti*>(parent_mach))) {
    // Maybe sub-machines will have the right mtype
    int nb_sub_mach = mach_multi->MachGetNb();
    for (int i=0; i<nb_sub_mach; i++) {
      EnqueueAllMatching(mtype, mach_multi->MachGet(i), queue);
    }
  }
}

// Returns all matching sub-machines in a vector.
std::vector<Mach*> FindAllMatching(int mtype, Mach* parent_mach)
{
  std::vector<Mach*> rval;
  EnqueueAllMatching(mtype, parent_mach, rval);
  return rval;
}
