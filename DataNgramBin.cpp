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
#include <sstream>
#include <string>

// system headers
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "Tools.h"
#include "Data.h"
#include "DataNgramBin.h"
#include "WordList.h"

const char* DATA_FILE_NGRAMBIN="DataNgramBin";
const int DATA_NGRAM_IGN_SHORT=1;	// ignore uncomplete n-grams 
					// if this options is not set, wholes will be filled with NULL_WORD
					// in order to simulate shorter n-grams
const int DATA_NGRAM_IGN_UNK=2;		// ignore n-grams with <UNK> at last position
const int DATA_NGRAM_IGN_UNKall=4;	// ignore n-grams that contain <UNK> anywhere
const int DATA_NGRAM_IGN_EOS=8;		// TODO: not implemented
const int DATA_NGRAM_IGN_ALL=15;


//*******************

void DataNgramBin::do_constructor_work()
{
    // parse header of binary Ngram file
  char full_fname[max_word_len]="";
  char full_SentSc_fname[max_word_len]="";

  if (path_prefix) {
    if (strlen(path_prefix)+strlen(fname)+2>(size_t)max_word_len)
      Error("full filename is too long");

    strcpy(full_fname, path_prefix);
    strcat(full_fname, "/");
  }
  strcat(full_fname, fname);

  if( (nb_SentSc > 0) ){
  	strcpy(full_SentSc_fname, path_prefix);
	strcat(full_SentSc_fname, "/");
	string rawname(fname);
	int fileExtPos = rawname.find_last_of(".");
	string rawfname = rawname.substr(0,fileExtPos+1);
	strcat(full_SentSc_fname,rawfname.c_str());
	strcat(full_SentSc_fname,SentSc_ext.c_str());
	SentScores = new double[nb_SentSc]; // vector of scores (double) when we deal with more than 1 Sentence score

     if(resampl_coeff < 1 ){
	cout<<" - reading resampling scores from file "<< full_SentSc_fname<<" (using "<<nb_SentSc<<" scores)"<<endl;
     }else{	     
  	cout<<" - SentenceScores activated but not used since resampl_coeff = "<<resampl_coeff<<endl;
  	}
     SentSc_fname = full_SentSc_fname; 	
  }

  fd=open(full_fname, O_RDONLY);
  if (fd<0) {
    perror(full_fname); Error();
  }
  read(fd, &id, sizeof(int));
  if (id>0) {
      // suppose old format without ID (we actually read the nbl)
    header_len=DATA_FILE_NGRAMBIN_HEADER_SIZE1;
    nbl=id; id=-1;
    read(fd, &nbex, sizeof(int));
    read(fd, &vocsize, sizeof(int));
  }
  else {
    switch (id) {
      case DATA_FILE_NGRAMBIN_VERSION2:
        header_len=DATA_FILE_NGRAMBIN_HEADER_SIZE2;
        read(fd, &nbl, sizeof(ulong));
        read(fd, &nbex, sizeof(ulong));
        read(fd, &vocsize, sizeof(WordList::WordIndex));
        break;
      default:
       ErrorN(" - %s binary ngram file: unsupported version %d\n",fname,-id);
    }
  }
 
  if( (nb_SentSc > 0) && (resampl_coeff < 1.0 ) ){
	SentSc_fs.open(SentSc_fname.c_str(), ios::in);
	CHECK_FILE(SentSc_fs, SentSc_fname.c_str());
  }
  int s;
  read(fd, &s, sizeof(int));
  if (s != sizeof(WordID)) {
    ErrorN("binary n-gram data uses %d bytes per index, but this code is compiled for %d byte indices\n", s, (int) sizeof(WordID));
  }
  read(fd, &bos, sizeof(WordID));
  read(fd, &eos, sizeof(WordID));
  read(fd, &unk, sizeof(WordID));
  printf(" - %s binary ngram file V%d with %lu words in %lu lines, order=%d, tgt_pos=%d, mode=%d (bos=%d, eos=%d, unk=%d)\n", fname, -id, nbex, nbl, order, tgpos, mode, bos, eos, unk);

  idim=order-1;
  odim=1;
 
  if (idim>0) {
    input = new REAL[idim + auxdim];
    wid = new WordID[order];
    if (! betweenSent_ctxt)	
      for (int i=0; i<order; i++) wid[i]= bos;
    else
      for (int i=0; i<order; i++) wid[i]= NULL_WORD;
    		    
  }
  target_vect = new REAL[odim];

    // initialize read buffer
  buf_n=0; buf_pos=-1;

  if (mode==0) { // we can directly get the numbers of n-grams from the header information
    nbs=0; nbw=nbex; nbu=nbi=0;
    nbex+=nbl;
    printf(" - %lu %d-grams (from header)\n", nbex, order);
  }
  else {
      // counting nbex to get true number of examples
    cout << "    counting ..."; cout.flush();
    time_t t_beg, t_end;
    time(&t_beg);

    ulong n=0;
    nbs=nbw=nbu=nbi=0;
    while (DataNgramBin::Next()) n++;

    time(&t_end);
    time_t dur=t_end-t_beg;
    printf(" %lu %d-grams (%lum%lus, %lu unk, %lu ignored)\n", n, order, dur/60, dur%60, nbu, nbi);

    if (n>nbex+nbl)
      Error("Number of counted examples is larger than the information in file header !?");
    nbex=n;  // 
  }
}

//*******************

DataNgramBin::DataNgramBin(char *p_prefix, ifstream &ifs, int p_aux_dim, const string& p_aux_ext, int p_nb_SentSc, const string& p_SentSc_ext, int p_betweenSentCtxt, DataNgramBin *prev_df)
 : DataFile::DataFile(p_prefix, ifs, p_aux_dim, p_aux_ext, p_nb_SentSc, p_SentSc_ext,p_betweenSentCtxt, prev_df),
  order(0), tgpos(0), eospos(0), mode(0), nbw(0), nbs(0), nbu(0), nbi(0)
{
  debug0("*** constructor DataNgramBin\n");
    // DataNgramBin <file_name> <resampl_coeff> <order> [<tgpos>] <mode>

    // parse addtl params -> 
  if (prev_df) {
    order=prev_df->order;	// use same order, tgpos and mode than previous datafiles
    tgpos=prev_df->tgpos;
    mode=prev_df->mode;	
  }
  else { //reading first DataFile followed by order, tgpos and mode
    ifs >> order >> tgpos >> mode;	// read order, tgpos and mode

    if( (ifs.rdstate() & std::ifstream::eofbit) || (ifs.rdstate() & std::ifstream::failbit ) ) {
      Error("Bad file format, should be DataNgramBin <file_name> <resampl_coef> <order> <tgpos> <mode> [factor_file1 factor_file2 ...]\n" );	
    }
  }
  if (order<2)
    Error("order must be at least 2\n");
  if (tgpos<0 || tgpos>=order)
    ErrorN("wrong value of target position: %d not in [0,%d]\n",tgpos,order-1);
  if (mode<0 || mode>DATA_NGRAM_IGN_ALL)
    Error("wrong value of DataNgramBin mode\n");

  do_constructor_work();
}

//*******************

DataNgramBin::DataNgramBin(char *p_fname, float p_rcoeff, int p_order)
  : DataFile::DataFile(NULL, p_fname, p_rcoeff),
    order(p_order), tgpos(p_order - 1), eospos(0), mode(3), nbw(0), nbs(0), nbu(0), nbi(0)
{
  debug0("*** constructor DataNgramBin with fname\n");

  do_constructor_work();
    // skip counting for efficieny reasons
  nbw=nbex+nbl; 	// this should be an upper bound on the number of n-grams
}

//*******************

DataNgramBin::DataNgramBin(char *p_fname, float p_rcoeff, int p_order, int p_tgpos, int p_mode)
  : DataFile::DataFile(NULL, p_fname, p_rcoeff),
    order(p_order), tgpos(p_tgpos), mode(p_mode), nbw(0), nbs(0), nbu(0), nbi(0)
{
  debug0("*** constructor DataNgramBin with fname\n");
  if (tgpos<0 || tgpos>=order)
    ErrorN("wrong value of target position: %d not in [0,%d]\n",tgpos,order-1);

  do_constructor_work();
    // skip counting for efficieny reasons
  nbw=nbex+nbl; 	// this should be an upper bound on the number of n-grams
}

//*******************

DataNgramBin::~DataNgramBin()
{
  debug0("*** destructor DataNgramBin\n");

  close(fd);
  if (idim>0) {
    delete [] wid;
    delete [] input;
  }
  delete [] target_vect;
}


//*******************
/*
 * Fill input and target_vec with data coming from the files
 * Factors are appended (not interleaved)
 * */
bool DataNgramBin::Next()
{
  //debug0("*** DataNgramBin::Next() \n");
  bool ok=false;
  string line_sc;
  int i;

   // we may need to skip some n-grams in function of the flags
  while (!ok) {

      // read from file into, return if EOF
    WordID w = NULL_WORD;
    // if eos word has been read in previous loop, insert NULL_WORD until encounter a NULL target or NULL context
    if ( (tgpos >= eospos) || (1 >= eospos) ) {
      if ( (DATA_FILE_BUF_SIZE>0) ) {
	      if (! ReadBuffered(&w)) return false;
      }
      else {
        if (read(fd, &w, sizeof(w)) != sizeof(w)) return false;
      }
    }
    
     // shift previous order
    for (i=1; i<order; i++) wid[i-1] = wid[i];
    wid[order-1]=w;

    if (0 < eospos)
      eospos--;

      // update statistics
    if (w == eos) {
      eospos = (order - 1);
      nbs++;
    }
    else if (w == unk) nbu++;
    else nbw++;

      // check if n-gram is valid according to the selected mode

    if (w == bos) {
        // If we don't want to consider between Sent context
	// new BOS, initialize the whole order to NULL_WORD
        // and terminate with BOS (it will be shifted away) 
      if (! betweenSent_ctxt ){
          for (i=0; i<order-1; i++) wid[i] = NULL_WORD;
          wid[i]=bos;
      }

        // new sentence --> read current sentence score	
      if(SentSc_fs.is_open()) {
	std::getline(SentSc_fs,line_sc);
	SentScores[0] = std::atof(line_sc.c_str() ); // TODO add support of multiple scores per line 
      }

        // read next line of auxiliary data
      if (aux_fs.is_open()) {
        for (i = 0; i < auxdim ; i++) {
          aux_fs >> input[idim + i];
          if (!aux_fs)
          {
	    cout << " - Error in auxiliary data file " << aux_fname << endl;
            Error("Not enough auxiliary data available");
	  }
        }
      }
      continue;
    }

    if (mode & DATA_NGRAM_IGN_UNK) {
        // ignore n-grams with <UNK> at last position
      if (w == unk) {
        nbi++;
        continue;
      }
    }

    if (mode & DATA_NGRAM_IGN_UNKall) {
        // ignore n-grams that contain <UNK> anywhere
      for (i=0; i<order-1; i++) {
        if (wid[i] == unk) {
          nbi++;
          break;
        }
      }
      if (i < order-1) continue;
    }

    if (mode & DATA_NGRAM_IGN_SHORT) {
        // ignore n-grams that contain NULL_WORD elsewhere than at 1st position
      for (i=0; i<order; i++)
        if (wid[i] == NULL_WORD) {
          nbi++;
          break;
      }
      if (i < order) continue;
    }

      /* standard mode */
    ok = (wid[tgpos] != NULL_WORD);	// when tgpos is different than last word we need to fill-up the n-gram until we have a word
    ok = (ok && (0 < tgpos) && (wid[tgpos - 1] != NULL_WORD)); // we need also to fill-up the n-gram until we have a word before target position
  } // of while (!ok)

  for (i=0; i<tgpos; i++) input[i] = (REAL) wid[i]; 	// careful: we cast to float which may give rounding problems of the integers
  for (; i<order-1; i++) input[i] = (REAL) wid[i+1];
  target_vect[0] = (int) wid[tgpos];
  target_id = (int) wid[tgpos];
  
  idx++;
  return true;
}

/********************
 *
 ********************/


void DataNgramBin::Rewind()
{
  debug0("*** DataNgramBin::Rewind()\n");
  lseek(fd, header_len, SEEK_SET);
  if (aux_fs.is_open())
    aux_fs.seekg(0, aux_fs.beg);
  if(SentSc_fs.is_open())
    SentSc_fs.seekg(0, aux_fs.beg);
  idx=-1;
  debug0("*** DataNgramBin::Rewind() done\n");
    // initialize read buffer
  buf_n=0; buf_pos=-1;
  eospos = 0;
}
 
