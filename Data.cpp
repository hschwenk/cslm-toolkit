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
 */

using namespace std;
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sstream>
#include <string>
#include <math.h>

#include "Tools.h"
#include "Data.h"
#include "DataAscii.h"
#include "DataAsciiClass.h"
#include "DataMnist.h"
#include "DataNgramBin.h"
#include "DataPhraseBin.h"

const char* DATA_HEADER_TXT="DataDescr";
const int   DATA_HEADER_ID=3;
const char* DATA_PRELOAD="Preload";
const int   DATA_PRELOAD_ACT=8;		// preloading is activated, additional flags:
const int   DATA_PRELOAD_TODO=1;	//    mark that preloading was not yet done, we use this to avoid multiple (costly) preloading
					//    this flag is set by Next() -> new Rewind() triggers resampling
const int   DATA_PRELOAD_ONCE=4;	//    we resample only once, even if rewind is called many times
const char* DATA_RESAMPL_MODE="ResamplMode";
const char* DATA_RESAMPL_SEED="ResamplSeed";
const char* DATA_SHUFFLE_MODE="ShuffleMode";
const char* DATA_NORMALIZE_MODE="Normalize";
const char* DATA_PATH_PREFIX="PathPrefix";
const char* DATA_NB_FACTORS="NumberFactors";
const char* DATA_SENTENCE_SCORES="SentenceScores";
const char* DATA_AUXILIARY_DATA="AuxiliaryData";
const char* DATA_BETWEEN_SENT_CONTEXT="BetweenSentenceContext";
const char* DATA_WORD_LIST="WordList";
const char* DATA_WORD_LIST_TARGET="WordListTarget";
const char* DATA_WORD_LIST_SOURCE="WordListSource";
const char* DATA_WORD_LIST_MULT="WordListMultiple";
const char* DATA_WORD_LIST_UNSTABLE="UseUnstableSort";	// switch for compatibility with CSLM <= V3.0 to use unstable sort
const char* DATA_WITH_EOS="with_eos";

/**************************
 *
 **************************/

Data::Data()
 : use_unstable_sort(false),
   fname(NULL), path_prefix(NULL), idim(0), odim(0), auxdim(0), nb_SentSc(0), tgpos(-1), nb_totl(0),
   preload(0),betweenSent_ctxt(0), resampl_mode(0), resampl_seed(1234567890), shuffle_mode(0), norm_mode(0), nb_factors(1),
   sr_wl_idx(-1), tg_wl_idx(-1), current_df(0), idx(-1),
   mem_cdf(NULL), mem_inp(NULL), mem_trg(NULL), input(NULL), target(NULL), aux(NULL)
{}

Data::Data(const char *p_fname, Data *other_data, bool use_class)
 : use_unstable_sort(false),
   fname(p_fname), path_prefix(NULL), idim(0), odim(0), auxdim(0), nb_SentSc(0), tgpos(-1), nb_totl(0),
   preload(0),betweenSent_ctxt(0), resampl_mode(0), resampl_seed(1234567890), shuffle_mode(0), norm_mode(0), nb_factors(1),
   sr_wl_idx(-1), tg_wl_idx(-1), current_df(0), idx(-1),
   mem_cdf(NULL), mem_inp(NULL), mem_trg(NULL), input(NULL), target(NULL), aux(NULL)
{
  debug0("* constructor Data\n");
  ReadFile(other_data, use_class);
}

void Data::ReadFile(Data *other_data, bool use_class)
{
  cout << "Opening data description '" << fname << "'" << endl;
  ifstream ifs;
  ifs.open(fname,ios::in);
  CHECK_FILE(ifs,fname);

    // parsing data description
  int i=ReadInt(ifs,DATA_HEADER_TXT);
  if (i>DATA_HEADER_ID) Error("unknown data description header\n");

  vector<ushort> lang_used; // languages used for each data file (1: only one, 2: source and target, -1: several)
  string auxdata_ext;
  string SentSc_ext;

  while (!ifs.eof()) {
    bool ok=false;
    vector<DataFile*> factored_df;

    string buf; char line[DATA_LINE_LEN];
    ifs >> buf;
    if (buf[0]=='#') {ifs.getline(line, DATA_LINE_LEN); continue;} // skip comments
    if (buf=="") break; // HACK
    if (buf==DATA_PRELOAD) { preload=DATA_PRELOAD_ACT | DATA_PRELOAD_TODO; ok=true; }
    if (buf==DATA_RESAMPL_MODE) { ifs >> resampl_mode; ok=true; }
    if (buf==DATA_RESAMPL_SEED) { ifs >> resampl_seed; ok=true; }
    if (buf==DATA_SHUFFLE_MODE) { ifs >> shuffle_mode; ok=true; }
    if (buf==DATA_NORMALIZE_MODE) { ifs >> norm_mode; ok=true; }
    if (buf==DATA_WORD_LIST_UNSTABLE) { use_unstable_sort=true; ok=true; }
    if (buf==DATA_PATH_PREFIX) {
      string tmp;
      ifs >> tmp; ok=true;
      cout << "Prefix for all data files: " << tmp << endl;
      path_prefix=strdup(tmp.c_str()); // ugly
    }
    if (buf==DATA_NB_FACTORS) {
      ifs >> nb_factors;
      if (nb_factors<1) Error("The number of factors must be at least one");
      ok=true;
    }
    if (buf==DATA_SENTENCE_SCORES) {
       string SentScInfo_buf;
       getline(ifs,SentScInfo_buf);
       stringstream SentScInfo_str(SentScInfo_buf);
       SentScInfo_str >> nb_SentSc >> SentSc_ext >> ExpGrowth;

       if(!SentScInfo_str){
	       nb_SentSc  = 1;
	       SentSc_ext = "scores";
       	       ExpGrowth  = 0.0; 	 	       
       }
       if(ExpGrowth < 0 ) ExpGrowth = 0;
       
       if( ExpGrowth )
	       cout<<"Resampling with ExpGrowth ("<<ExpGrowth<<") using "<<nb_SentSc<<" Sentences Scores extracted from "<<SentSc_ext<<" files (same name and location as data files)"<<endl;
       else
	       cout<<"Resampling without ExpGrowth using "<<nb_SentSc<<" Sentences Scores extracted from"<<SentSc_ext<<" files (same name and location as data files)"<<endl;

      ok=true;
    }
     if (buf==DATA_BETWEEN_SENT_CONTEXT) {
	     cout<<"Using continuous context ..." <<endl;	
	     betweenSent_ctxt=1;
             ok=true;
     } // Fethi For between Sentences ctxt
     if (buf==DATA_AUXILIARY_DATA) {
      string dad_buf;
      getline(ifs, dad_buf);
      stringstream dad_str(dad_buf);
      dad_str >> auxdim >> auxdata_ext; // read dimension and file extension
      if (!dad_str)
        auxdata_ext = "aux"; // use default extension
      else if ('.' == auxdata_ext[0])
        auxdata_ext.erase(0, 1);
      if (auxdim<1) Error("The auxiliary data dimension must be at least one");
      ok=true;
    }
    if (buf==DATA_WORD_LIST_SOURCE) {
      sr_wl_idx = wlist.size();
      CreateWordList(ifs, sr_wl_idx, use_class);
      ok=true;
    }
    if (   (buf==DATA_WORD_LIST       )
        || (buf==DATA_WORD_LIST_TARGET) ) {
      tg_wl_idx = wlist.size();
      CreateWordList(ifs, tg_wl_idx, use_class);
      ok=true;
    }
    if (buf==DATA_WORD_LIST_MULT) {
      size_t idx = -1;
      ifs >> idx;
      if (idx != (size_t)-1) {
        CreateWordList(ifs, idx, use_class);
        ok=true;
      }
    }

    if (buf==DATA_FILE_ASCII) {
      factored_df.clear();
      factored_df.push_back(new DataAscii(path_prefix,ifs, auxdim, auxdata_ext, nb_SentSc, SentSc_ext, betweenSent_ctxt));
      for (int i=1; i<nb_factors; i++)
        factored_df.push_back(new DataAscii(path_prefix,ifs, auxdim, auxdata_ext, nb_SentSc, SentSc_ext, betweenSent_ctxt, (DataAscii*)factored_df[0]));
      datafile.push_back(factored_df);
      lang_used.push_back(2);
      ok=true;
    }

    if (buf==DATA_FILE_ASCIICLASS) {
      factored_df.clear();
      factored_df.push_back(new DataAsciiClass(path_prefix,ifs, auxdim, auxdata_ext, nb_SentSc, SentSc_ext, betweenSent_ctxt));
      for (int i=1; i<nb_factors; i++)
        factored_df.push_back(new DataAsciiClass(path_prefix,ifs, auxdim, auxdata_ext, nb_SentSc, SentSc_ext,betweenSent_ctxt, (DataAsciiClass*)factored_df[0]));
      datafile.push_back(factored_df);
      lang_used.push_back(1);
      ok=true;
    }

    if (buf==DATA_FILE_MNIST) {
      factored_df.clear();
      factored_df.push_back(new DataMnist(path_prefix,ifs, auxdim, auxdata_ext, nb_SentSc, SentSc_ext,betweenSent_ctxt));
      for (int i=1; i<nb_factors; i++)
        factored_df.push_back(new DataMnist(path_prefix,ifs, auxdim, auxdata_ext, nb_SentSc, SentSc_ext,betweenSent_ctxt, (DataMnist*)factored_df[0]));
      datafile.push_back(factored_df);
      lang_used.push_back(1);
      ok=true;
    }

    if (buf==DATA_FILE_NGRAMBIN) {
      factored_df.clear();
      DataNgramBin* dataNgramBin;
      factored_df.push_back(dataNgramBin = new DataNgramBin(path_prefix,ifs, auxdim, auxdata_ext, nb_SentSc, SentSc_ext,betweenSent_ctxt));
      if (NULL != dataNgramBin)
        tgpos = dataNgramBin->GetTgPos();
      for (int i=1; i<nb_factors; i++){  //TODO: this does not work since getline is called in DataNgramBin constructor -> ifs is no more correct !!!
	  //cerr << "reading " << nb_factors-1 << " factor datafiles ..." << endl;
	  factored_df.push_back(new DataNgramBin(path_prefix,ifs, auxdim, auxdata_ext, nb_SentSc, SentSc_ext,betweenSent_ctxt, (DataNgramBin*)factored_df[0]));
      }
      datafile.push_back(factored_df);
      lang_used.push_back(1);
      ok=true;
    }


    if (buf==DATA_FILE_PHRASEBIN) {
      factored_df.clear();
      for (int i = 0 ; i < nb_factors ; i++) {
        factored_df.push_back(new DataPhraseBin(path_prefix, ifs, auxdim, auxdata_ext, nb_SentSc, SentSc_ext,betweenSent_ctxt, (0 < i) ? (DataPhraseBin*)factored_df[0] : NULL));
        if ((sr_wl_idx != (size_t)-1) && (tg_wl_idx != (size_t)-1))
          factored_df.back()->SetWordLists(&(wlist[sr_wl_idx]->at(i)), &(wlist[tg_wl_idx]->at(i)));
      }
      datafile.push_back(factored_df);
      lang_used.push_back(2);
      ok=true;
    }

    /*if (datafile.size()==1) {
        // input and output dimension is sum of factors
      idim=odim=0;
      for (vector<DataFile*>::iterator it=datafile[0].begin(); it!=datafile[0].end(); ++it) {
        idim+=(*it)->GetIdim();
        odim+=(*it)->GetOdim();
      }
    }*/
    if (datafile.size()>=1) {
        // check whether additional datafiles have the same dimensions (this is implicitly OK for factors)
	// Loic -> why not checking factors also ?
        // TODO: check nb of examples
      idim=odim=0;
      for(vector< vector<DataFile*> >::iterator itdf=datafile.begin(); itdf!=datafile.end(); ++itdf){
	  int nidim=0, nodim=0;
	  for (vector<DataFile*>::iterator itfactor=(*itdf).begin(); itfactor!=(*itdf).end(); ++itfactor) {
	    nidim+=(*itfactor)->GetIdim();
	    nodim+=(*itfactor)->GetOdim();
	  }
	  if(idim==0 && odim==0){
	    idim=nidim; odim=nodim;
	  } else {
	    if (idim != nidim) Error("Data::Readfile: mismatch in input dimension\n");
	    if (odim != nodim) Error("Data::ReadFile: mismatch in output dimension\n");
	  }
      }
    }

    if (!ok) {
      ifs.getline(line, DATA_LINE_LEN);
      cerr << buf << "" << line << endl;
      Error("Data::ReadFile: parse error in above line of the datafile\n");
    }
  }
  ifs.close();
  if (0 > tgpos)
    // set default target position
    tgpos = idim;

  // check word lists
  if (tg_wl_idx == (size_t)-1) {
    for (vector<ushort>::const_iterator ci = lang_used.begin(), cie = lang_used.end() ; ci != cie ; ci++)
      if ((1 == *ci) || (2 == *ci)) // target word list is needed
        Error("No target word list given\n");
  }
  else if (other_data != NULL) {
    size_t stNbWList = min(wlist.size(), other_data->wlist.size());
    for (size_t st = 0 ; st < stNbWList ; st++) {
      if (st == sr_wl_idx)
        CompareWordList(sr_wl_idx, *other_data, other_data->sr_wl_idx);
      else if ((st == tg_wl_idx) && (sr_wl_idx != (size_t)-1))
        CompareWordList(tg_wl_idx, *other_data, other_data->tg_wl_idx);
      else
        CompareWordList(st, *other_data, st);
    }
  }

  nb_totl=0;
  cout << "Summary of used data: (" << nb_factors << " factors)" << endl;
  df_dim.resize(datafile.size());
  for (size_t df = 0, dfs = datafile.size() ; df < dfs ; df++) {
    DataFile* dff = datafile[df].front();
    nb_totl+=dff->Info();
    if (nb_factors>1) {
      for (i=1; i<nb_factors; i++) {
        printf("    f%d: ",i+1);
        datafile[df][i]->Info("");
      }
    }
    df_dim[df].resize(wlist.size(), make_pair<int, REAL*>(0, NULL));
    switch(lang_used[df]) {
    case 2:
      df_dim[df][1].first = dff->GetOdim();
      df_dim[df][1].second = dff->target_vect;
    case 1:
      df_dim[df][0].first = dff->GetIdim();
      df_dim[df][0].second = dff->input;
      break;
    }
  }

  cout << " - total number of examples: " << nb_totl << endl;
  cout << " - dimensions: input=" << idim << ", output=" << odim << endl;
  if (resampl_mode) {
    cout << " - resampling with seed " << resampl_seed << endl;
    srand48(resampl_seed);
  }
  if (preload > 0) {
    printf(" - allocating preload buffer of %.1f GBytes\n", (REAL) ((size_t) nb_totl*idim*sizeof(REAL) / 1024 / 1024 / 1024));
    mem_cdf = new int[nb_totl];
    mem_inp = new REAL[(size_t) nb_totl*idim];	// cast to 64bit !
    if (odim>0) mem_trg = new REAL[(size_t) nb_totl*odim];

      // check whether there is a resampling coeff != 0
      // i.e. we need to resample at each rewind
    double s = 0.0;
    for (FactoredDataFiles::iterator itf = datafile.begin(); itf!=datafile.end(); ++itf)
      s+=(*itf)[0]->GetResamplCoef();
    if (s>=datafile.size()) {
      preload|=DATA_PRELOAD_ONCE;
      cout << " - all resampling coefficients are set to one, loading data once\n";
    }

  }
  else {
    if (norm_mode>0)
      Error("Normalization of the data is only implemented with preloading\n");
  }
  Preload();
  Shuffle();
}

/**************************
 *
 **************************/

Data::~Data()
{
  debug0("* destructor Data\n");
  if (preload) {
    delete [] mem_cdf;
    delete [] mem_inp;
    if (odim>0) delete [] mem_trg;
  }
  for (FactoredDataFiles::iterator itf = datafile.begin(); itf!=datafile.end(); ++itf)
    for (vector<DataFile*>::iterator it = (*itf).begin(); it!=(*itf).end(); ++it)
      delete (*it);
  datafile.clear();
  for (size_t st = 0 ; st < wlist.size() ; st++)
    DeleteWordList(wlist[st], w_shared[st], w_mutex[st]);
  wlist.clear();
  w_mutex.clear();
  w_shared.clear();
  if (path_prefix) free(path_prefix);
}


/**************************
 *
 **************************/

void Data::Shuffle()
{
  if (shuffle_mode < 1 || !preload) return;

  time_t t_beg, t_end;
  time(&t_beg);

  REAL	*inp = new REAL[idim];
  REAL	*trg = new REAL[odim];

  cout << " - shuffling data " << shuffle_mode << " times ...";
  cout.flush();
  for (ulong i=0; i<shuffle_mode*(ulong)nb_totl; i++) {
    ulong i1 = (ulong) (nb_totl * drand48());
    ulong i2 = (ulong) (nb_totl * drand48());

    int cdf = mem_cdf[i1];
    mem_cdf[i1] = mem_cdf[i2];
    mem_cdf[i2] = cdf;

    memcpy(inp, mem_inp + i1*idim, idim*sizeof(REAL));
    memcpy(mem_inp + i1*idim, mem_inp + i2*idim, idim*sizeof(REAL));
    memcpy(mem_inp + i2*idim, inp, idim*sizeof(REAL));

    if (odim>0) {
      memcpy(trg, mem_trg + i1*odim, odim*sizeof(REAL));
      memcpy(mem_trg + i1*odim, mem_trg + i2*odim, odim*sizeof(REAL));
      memcpy(mem_trg + i2*odim, trg, odim*sizeof(REAL));
    }

  }

  delete [] inp; delete [] trg;

  time(&t_end);
  time_t dur=t_end-t_beg;

  cout << " done (" << dur / 60 << "m" << dur % 60 << "s)" << endl;
}

//**************************
//
//
/*
 * Preload: read datafiles and put the content into mem_inp and mem_trg
 * Factors are appended (not interleaved)
 * 
 * */
void Data::Preload()
{
  if (!preload) return;
  if (! (preload&DATA_PRELOAD_TODO)) {
    cout << " - all data is already loaded into memory" << endl;
    return;
  }
  preload &= ~DATA_PRELOAD_TODO; // clear flag

  cout << " - loading all data into memory ...";
  ++Epoch_num;
  cout.flush();
  time_t t_beg, t_end;
  time(&t_beg);

    // rewind everything
  for (FactoredDataFiles::iterator itf = datafile.begin(); itf!=datafile.end(); ++itf) {
    for (vector<DataFile*>::iterator it = (*itf).begin(); it!=(*itf).end(); ++it) (*it)->Rewind();
  }

  int idx=0;
  int cdf=0;
  for (FactoredDataFiles::iterator itf = datafile.begin(); itf!=datafile.end(); ++itf, ++cdf) {

      // get the required number of examples from all factors
    int n = -1, maxn = (*itf)[0]->GetNbresampl();
    int idim1=(*itf)[0]->GetIdim();  // dimension of one factor (identical for all, e.g. a 7-gram)
    int odim1=(*itf)[0]->GetOdim();

    while (++n < maxn) {
      debug1("getting example %d\n",idx);
      mem_cdf[idx] = cdf;

      bool ok=false;
      while (!ok) {
          // advance synchronously all factors until ok
        for (vector<DataFile*>::iterator it = (*itf).begin(); it!=(*itf).end(); ++it) {
          debug1("  next factor %ld\n", it-(*itf).begin());
          if (! (*it)->Next()) (*it)->Rewind(); // TODO: deadlock if file empty
        }

	if( ((*itf)[0]->GetNB_SentScores() > 0) && ((*itf)[0]->GetResamplCoef() < 1.0 )  ){
		ok = (drand48() <  ( (*itf)[0]->GetResamplScore() * (exp(- (float)ExpGrowth/Epoch_num)) ) );
	}else{
		ok = (drand48() < (*itf)[0]->GetResamplCoef());
	}
	
	debug1("  %s\n", ok ? "keep" : "skip");
      }

        // copy all factors sequentially in memory
      REAL *adr_inp=mem_inp + (size_t) idx*idim;
      REAL *adr_trg=mem_trg + (size_t) idx*odim;
      for (vector<DataFile*>::iterator it = (*itf).begin(); it!=(*itf).end(); ++it) {
        debug2("  load factor %ld to address %p\n", it-(*itf).begin(), adr_inp);
        memcpy(adr_inp, (*it)->input, idim1*sizeof(REAL));
	adr_inp+=idim1;
        if (odim1 > 0) {
          memcpy(adr_trg, (*it)->target_vect, odim1*sizeof(REAL));
	  adr_trg+=odim1;
        }
      }
      idx++; // next example
    }

  }

  if (norm_mode & 1) {
    cout << " subtract mean,"; cout.flush();
    for (int i=0; i<idim; i++) {
      int e;
      REAL m=0, *mptr;
      for (e=0, mptr=mem_inp+i; e<idx; e++, mptr+=idim) m+=*mptr;
      m = m/idx; // mean
      debug2("mean[%d]=%f\n", i, m);
      for (e=0, mptr=mem_inp+i; e<idx; e++, mptr+=idim) *mptr -= m;
    }
  }

  if (norm_mode & 2) {
    cout << " divide by variance,"; cout.flush();
    for (int i=0; i<idim; i++) {
      int e;
      REAL m=0, m2=0, *mptr;
      for (e=0, mptr=mem_inp+i; e<idx; e++, mptr+=idim) { m+=*mptr; m2+=*mptr * *mptr; }
      m = m/idx;  // mean
      m2 = m2/idx - m; // var = 1/n sum_i x_i^2  -  mu^2
      debug3(" mean, var[%d]=%f %f\n", i, m, m2);
      if (m2>0)
        for (e=0, mptr=mem_inp+i; e<idx; e++, mptr+=idim)
          *mptr = (*mptr - m) / m2;
    }
  }
#ifdef DEBUG
    printf("DUMP PRELOADED DATA at adr %p (%d examples of dim %d->%d):\n",mem_inp,idx,idim,odim);
    for (int e=0; e<idx; e++) {
      for (int i=0; i<idim; i++) printf(" %5.2f",mem_inp[(size_t) e*idim+i]); printf("\n");
    }
#endif

  time(&t_end);
  time_t dur=t_end-t_beg;
  cout << " done (" << dur / 60 << "m" << dur % 60 << "s)" << endl;
}


/**************************
 *
 **************************/

void Data::Rewind()
{
  debug0("** Data::Rewind()\n");
  if (preload) {
       // clear all data, resample and shuffle again
    Preload();
    Shuffle();
  }
  else {
    for (FactoredDataFiles::iterator itf = datafile.begin(); itf!=datafile.end(); ++itf)
      for (vector<DataFile*>::iterator it = (*itf).begin(); it!=(*itf).end(); ++it)
        (*it)->Rewind();
  }
  idx = -1;
  debug0("** Data::Rewind() done\n");
}

/**************************
 * Advance to next data
 **************************/
/* 
 * set 'input' and 'target' pointers to the next values (pointing into mem_inp and mem_trg)
 * */
bool Data::Next()
{
  if (idx >= nb_totl-1) return false;
  idx++;

  if (preload) {
      // just advance to next data in memory
    input = &mem_inp[(size_t) idx*idim];
    aux = (input + (idim - auxdim));
    if (odim>0) target = &mem_trg[(size_t) idx*odim];
    current_df = mem_cdf[idx];

    // handling multiple languages
    const size_t nb_lang = df_dim[current_df].size();
    switch (nb_lang) {
    default: {
        REAL* cur_input = input;
        for (size_t i = 0 ; nb_lang > i ; i++) {
          df_dim[current_df][i].second = cur_input;
          cur_input += df_dim[current_df][i].first;
        }
      }
      break;
    case 2:
      df_dim[current_df][1].second = target;
    case 1:
      df_dim[current_df][0].second = input;
      break;
    }
//printf("DATA:"); for (int i =0; i<idim; i++) printf(" %5.2f", input[i]); printf("\n");
    if (!(preload&DATA_PRELOAD_ONCE)) preload |= DATA_PRELOAD_TODO; // remember that next Rewind() should preload again
    return true;
  }

  if (nb_factors>1)
    Error("multiple factors are only implemented with preloading");

  if (shuffle_mode > 0) {
      // resample in RANDOMLY SELECTED datafile until data was found
      // we are sure to find something since idx was checked before
    current_df = (int) (drand48() * datafile.size());
//cout << " df=" << df << endl;
    datafile[current_df][0]->Resampl();
    input = datafile[current_df][0]->input;
    if (odim>0) target = datafile[current_df][0]->target_vect;
  }
  else {
      // resample SEQUENTIALLY all the data files
    static int i=-1, nbdf=datafile[current_df][0]->GetNbex();
    if (idx==0) {current_df = 0, i=-1, nbdf=datafile[current_df][0]->GetNbex(); }	// (luint) this) is a hack to know when there was a global rewind
    if (++i >= nbdf) { current_df++; nbdf=datafile[current_df][0]->GetNbex(); i=-1; }
    if (current_df >= (int) datafile.size()) Error("internal error: no examples left\n");
//printf("seq file: current_df=%d, i=%d\n", current_df,i);
    datafile[current_df][0]->Resampl();	//TODO: idx= ??
//cout << " got df=" << df << " idx="<<idx<<endl;
    input = datafile[current_df][0]->input;
    if (odim>0) target = datafile[current_df][0]->target_vect;
  }
  aux = (input + (idim - auxdim));

  return true;
}

//**************************
//
//

void Data::CreateWordList(ifstream &ifs, size_t idx, bool use_class)
{
  // resize vectors
  if (wlist.size() <= idx) {
    size_t ns = (idx + 1);
    wlist   .resize(ns, NULL);
    w_shared.resize(ns, NULL);
    w_mutex .resize(ns, NULL);
  }
  vector<WordList> *&iw = wlist[idx];
  pthread_mutex_t *&im = w_mutex[idx];
  int *&is = w_shared[idx];

  if (im != NULL)
    pthread_mutex_lock(im);

  // new word list
  if (iw == NULL)
    iw = new vector<WordList>;
  if (iw == NULL)
    Error("Can't allocate word list");
  iw->reserve(nb_factors);
  iw->resize(nb_factors);
  vector<string> vsPath(nb_factors, (NULL != path_prefix) ? (string(path_prefix) += '/') : string());
  stringbuf sb;
  ifs.get(sb);
  istream istr(&sb);
  for (int i=0; i<nb_factors; i++) {
    string fname;
    istr >> fname;
    vsPath[i] += fname;
  }
  string buf;
  istr >> buf;
  bool bUseEos = (DATA_WITH_EOS == buf);
  for (int i=0; i<nb_factors; i++) {
    cout << " - reading word list from file " << vsPath[i] << flush;
    iw->at(i).SetSortBehavior(!use_unstable_sort);
    WordList::WordIndex voc_size = iw->at(i).Read(vsPath[i].c_str(), use_class, bUseEos);
    cout << ", got " << voc_size << " words" << endl;
  }

  if (im != NULL)
    pthread_mutex_unlock(im);
  else {
    // word list sharing
    im = new pthread_mutex_t;
    if (im != NULL) {
      pthread_mutex_init(im, NULL);
      int *new_is = new int;
      if (new_is != NULL) {
        (*new_is) = 0;
        is = new_is;
      }
    }
  }
}

void Data::CompareWordList(size_t ii, Data &other_data, size_t ei)
{
  if ((ii >= this->wlist.size()) || (ei >= other_data.wlist.size()))
    return;
  vector<WordList> *&iw = this->wlist[ii];
  vector<WordList> * ew = other_data.wlist[ei];
  pthread_mutex_t *&im = this->w_mutex[ii];
  pthread_mutex_t * em = other_data.w_mutex[ei];
  int *&is = this->w_shared[ii];
  int * es = other_data.w_shared[ei];
  if ((iw == NULL) || (ew == NULL))
    return;

  // compare with other word list
  size_t stCurWl = 0;
  size_t stWlCountI = iw->size();
  size_t stWlCountE = ew->size();
  if (stWlCountI == stWlCountE)
    for (; stCurWl < stWlCountI ; stCurWl++) {
      WordList::WordIndex wiCur = 0;
      WordList::WordIndex wiSize = (*iw)[stCurWl].GetSize();
      if (wiSize != (*ew)[stCurWl].GetSize())
        break;
      for (; wiCur < wiSize ; wiCur++) {
        WordList::WordInfo &wiInt = (*iw)[stCurWl].GetWordInfo(wiCur);
        WordList::WordInfo &wiExt = (*ew)[stCurWl].GetWordInfo(wiCur);
        if ((wiInt.id != wiExt.id) || (wiInt.n != wiExt.n)  || (wiInt.cl != wiExt.cl)
            || (strcmp(wiInt.word, wiExt.word) != 0)    )
          break;
      }
      if (wiCur < wiSize)
        break;
    }
  if ((stCurWl < stWlCountI) || (stCurWl < stWlCountE))
    Error("Word lists are not identical\n");
  else {
    vector<WordList> *old_iw = iw;
    pthread_mutex_t *old_im = im;
    int *old_is = is;

    // share other word list
    int inc_is = 0;
    if (em != NULL) {
      pthread_mutex_lock(em);
      inc_is = ((es != NULL) ? (*es) + 1 : 0);
      if (inc_is > 0) {
        (*es) = inc_is;
        iw = ew;
        is = es;
        im = em;
      }
      pthread_mutex_unlock(em);
    }
    if (inc_is <= 0)
      Error ("Can't share word list\n");
    else
      // remove previous word list
      DeleteWordList(old_iw, old_is, old_im);
  }
}

void Data::DeleteWordList(vector<WordList> *&iw, int *&is, pthread_mutex_t *&im)
{
  vector<WordList> *old_iw = iw;
  pthread_mutex_t *old_im = im;
  int *old_is = is;
  is = NULL;
  iw = NULL;
  im = NULL;

  // verify if word list is shared
  if (old_im != NULL) {
    pthread_mutex_lock(old_im);
    if (old_is != NULL) {
      if ((*old_is) > 0) {
        (*old_is)--;
        pthread_mutex_unlock(old_im);
        return;
      }
      else
        delete old_is;
    }
  }

  if (old_iw != NULL)
    delete old_iw;

  // destroy mutex
  if (old_im != NULL) {
    pthread_mutex_unlock(old_im);
    pthread_mutex_destroy(old_im);
    delete old_im;
  }
}
