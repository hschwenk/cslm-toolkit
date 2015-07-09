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

// system headers
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#include "Tools.h"
#include "DataPhraseBin.h"

const char* DATA_FILE_PHRASEBIN="DataPhraseBin";
const int DATA_PHRASE_IGN_SHORT_SRC=1;		// ignore phrase pairs for which the source phrase is to short
const int DATA_PHRASE_IGN_SHORT_TGT=16;		// ignore phrase pairs for which the target phrase is to short
const int DATA_PHRASE_IGN_ALL=17;		// all flags ORed together
const int max_buf_len=32;			// maximum length of phrases that can be read


//*******************

void DataPhraseBin::do_constructor_work()
{
  ulong n;

  char full_fname[max_word_len]="";

  if (path_prefix) {
    if (strlen(path_prefix)+strlen(fname)+2>(size_t)max_word_len)
      Error("full filename is too long");

    strcpy(full_fname, path_prefix);
    strcat(full_fname, "/");
  }
  strcat(full_fname, fname);

    // parse header binary Ngram file

  fd=open(full_fname, O_RDONLY);
  if (fd<0) {
    perror(full_fname); Error();
  }
  int s;
  read(fd, &s, sizeof(int));
  if (s != sizeof(WordID)) {
    ErrorN("binary phrase data uses %d bytes per index, but this code is compiled for %d byte indices\n", s, (int) sizeof(WordID));
  }

  read(fd, &max_len, sizeof(int));		// maximal length of a phrase (source or target)
  if (max_len<1 || max_len>255) {
    ErrorN("binary phrase data: max length must be in 1..255\n");
  }

    // source side vocabulary infos
  read(fd, &ivocsize, sizeof(int));
  read(fd, &ibos, sizeof(WordID));	// BOS
  read(fd, &ieos, sizeof(WordID));	// EOS, not used
  read(fd, &iunk, sizeof(WordID));	// UNK
  iempty= NULL_WORD; // used to identify empty words in phrase

    // read source counts
  inbphw = new int[max_len+1]; inbphw[0]=0;
  for (s=1; s<=max_len; s++) {read(fd, inbphw+s, sizeof(int)); inbphw[0]+=inbphw[s]; }
  printf(" - %s binary phrase pairs with %d entries of max length of %d, mode=%d\n", fname, inbphw[0], max_len, mode);

    // calc source cumulated counts
  icnbphw = new int[max_len+1];
  icnbphw[1]=inbphw[1];
  for (s=2; s<=max_len; s++) icnbphw[s]=icnbphw[s-1]+inbphw[s];
  printf("   source: vocabulary of %d words (bos=%d, eos=%d, unk=%d, empty=%d)\n", ivocsize, ibos, ieos, iunk, iempty);
 
    // target side vocabulary infos
  read(fd, &ovocsize, sizeof(int));
  read(fd, &obos, sizeof(WordID));
  read(fd, &oeos, sizeof(WordID));
  read(fd, &ounk, sizeof(WordID));
  oempty= NULL_WORD; // used to identify empty words in phrase
  printf("   target: vocabulary of %d words (bos=%d, eos=%d, unk=%d, empty=%d)\n", ovocsize, obos, oeos, ounk, oempty);

    // read target counts
  onbphw = new int[max_len+1]; onbphw[0]=0;
  for (s=1; s<=max_len; s++) {read(fd, onbphw+s, sizeof(int)); onbphw[0]+=onbphw[s]; }
  if (onbphw[0] != inbphw[0]) {
    ErrorN("number of source phrase (%d) does not match the number of target phrases (%d)\n", inbphw[0], onbphw[0]);
  }

    // calc target cumulated counts
  ocnbphw = new int[max_len+1];
  ocnbphw[1]=onbphw[1];
  for (s=2; s<=max_len; s++) ocnbphw[s]=ocnbphw[s-1]+onbphw[s];
 
  idim=src_phlen;
  odim=tgt_phlen;
 
  if (idim>0) {
    input = new REAL[idim + auxdim];
  }

  if (odim>0) {
    target_vect = new REAL[odim];
  }

    // initialize read buffer
  buf_n=0; buf_pos=-1;

  cout << "    statistics:" << endl;
  printf("       source:"); for (s=0; s<=max_len; s++) printf("%10d", inbphw[s]); printf("\n");
  printf("       target:"); for (s=0; s<=max_len; s++) printf("%10d", onbphw[s]); printf("\n");

  if (mode==0 && src_phlen==max_len && tgt_phlen==max_len) {
      // we will use all the data -> we can directly get the numbers of phrase pairs from the header information
    nbi=0;
    nbex=inbphw[0];
    printf(" - %lu phrase pairs of full length (from header)\n", nbex);
    return;
  }

  printf("   limiting phrase pairs to length %d/%d words, mode %d\n", src_phlen, tgt_phlen, mode);
  if (src_phlen == tgt_phlen) {
      // we can get an UPPER BOUND of the nbr of phrases directly from the file header
    n = icnbphw[src_phlen] < ocnbphw[tgt_phlen] ? icnbphw[src_phlen] : ocnbphw[tgt_phlen];
    nbi=inbphw[0]-n;
    printf("   header: upper bound of %lu phrase pairs (%d=%5.2f%% ignored)\n", n, nbi, 100.0*nbi/inbphw[0]);
  }
  

    // counting nbex to get true number of examples
  cout << "    counting ..."; cout.flush();
  time_t t_beg, t_end;
  time(&t_beg);

  int nbi=0; n=0;
  while (DataPhraseBin::Next()) n++;
  nbi=inbphw[0]-n;

  time(&t_end);
  time_t dur=t_end-t_beg;
  printf(" %lu phrase pairs (%lum%lus, %d=%5.2f%% ignored)\n", n, dur/60, dur%60, nbi, 100.0*nbi/inbphw[0]);

  if (n>(ulong)inbphw[0])
    Error("Number of counted examples is larger than the information in file header !?");
  nbex=n;
}

//*******************

DataPhraseBin::DataPhraseBin(char *p_prefix, ifstream &ifs, int p_aux_dim, const string& p_aux_ext, int p_nb_SentSc, const string& p_SentSc_ext, int p_betweenSentCtxt, DataPhraseBin* prev_df)
 : DataFile(p_prefix, ifs, p_aux_dim, p_aux_ext, p_nb_SentSc,p_SentSc_ext,p_betweenSentCtxt, prev_df),
 max_len(0), mode(0), src_phlen(0), tgt_phlen(0),
 iwlist(NULL), inbphw(NULL), icnbphw(NULL),
 owlist(NULL), onbphw(NULL), ocnbphw(NULL),
 nbi(0)
{
  debug0("*** constructor DataPhraseBin\n");
    // DataPhraseBin <file_name> <resampl_coeff> <src_phrase_len> <tgt_phrase_len> [flags]
    // parse addtl params
  if (prev_df) {
    src_phlen=prev_df->src_phlen;
    tgt_phlen=prev_df->tgt_phlen;
    mode=prev_df->mode;
  }
  else {
    ifs >> src_phlen >> tgt_phlen >> mode;
    if (src_phlen<1 || src_phlen>256)
      Error("length of source phrases must be in [1,256]\n");
    if (tgt_phlen<1 || tgt_phlen>256)
      Error("length of target phrases must be in [1,256]\n");
    if (mode<0 || mode>DATA_PHRASE_IGN_ALL)
      Error("wrong value of DataPhraseBin mode\n");
  }

  do_constructor_work();
}

//*******************

DataPhraseBin::DataPhraseBin(char *p_fname, float p_rcoeff, int p_src_phlen, int p_tgt_phlen, int p_mode)
  : DataFile::DataFile(NULL, p_fname, p_rcoeff),
  mode(p_mode), src_phlen(p_src_phlen), tgt_phlen(p_tgt_phlen), 
  iwlist(NULL), inbphw(NULL), icnbphw(NULL),
  owlist(NULL), onbphw(NULL), ocnbphw(NULL)
{
  debug0("*** constructor DataPhraseBin with fname\n");

  do_constructor_work();
    // TODO:  counting ?
}

//*******************

DataPhraseBin::~DataPhraseBin()
{
  debug0("*** destructor DataPhraseBin\n");

  close(fd);
  if (idim>0) delete [] input;
  if (odim>0) delete [] target_vect;
  if (inbphw) delete [] inbphw;
  if (icnbphw) delete [] icnbphw;
  if (onbphw) delete [] onbphw;
  if (ocnbphw) delete [] ocnbphw;
}

//*******************

void DataPhraseBin::SetWordLists(WordList *p_iwlist, WordList *p_owlist)
{
  iwlist=p_iwlist; owlist=p_owlist;
  if (iwlist->HasEOS()) {
    iempty=iwlist->GetEOSIndex();
    printf (" - source word list uses special word (%d) for short phrases\n",iempty);
  }
  if (owlist->HasEOS()) {
    oempty=owlist->GetEOSIndex();
    printf (" - target word list uses special word (%d) for short phrases\n",oempty);
  }
  
}

//*******************

bool DataPhraseBin::Next()
{
  //debug0("*** DataPhraseBin::Next() \n");
  bool ok=false;
  WordID buf[max_buf_len];

   // we may need to skip some phrase pairs in function of their length
  while (!ok) {
    int i;
    uchar src_len, tgt_len;

      // read source phrase
    if (!ReadBuffered(&src_len, sizeof(src_len))) return false;

    debug1("source read %d words:", src_len);
    if ((int) src_len>max_buf_len) Error("The source phrase is too long, you need to recompile the program\n");
    if (!ReadBuffered((uchar*)buf, src_len*sizeof(WordID))) Error("DataPhraseBin::Next(): no source phrase left\n");
#ifdef DEBUG
    for (i=0; i<src_len; i++) printf(" %d", buf[i]);
    printf("\n");
#endif
    if ((int) src_len>src_phlen) {
       debug0(" src too long -> flag to ignore\n");
       nbi++; // ignore: too many source words
       ok=false; // won't be used, but we still need to read the target phrase to keep it in sync
    }
    else {
        // copy source phrase into input vector
      for (i=0; i<src_len; i++) input[i] = (REAL) buf[i]; 	// careful: we cast int to float
      for (   ; i<src_phlen; i++) input[i] = (REAL) iempty;
      ok=true;
    }

      // read target phrase
    if (!ReadBuffered(&tgt_len, sizeof(tgt_len))) return false;
    debug1("target read %d words:", tgt_len);
    if ((int)tgt_len>max_buf_len) Error("The target phrase is too long, you need to recompile the program\n");
    if (!ReadBuffered((uchar*)buf, tgt_len*sizeof(WordID))) Error("DataPhraseBin::Next(): no target phrase left\n");
#ifdef DEBUG
    for (i=0; i<tgt_len; i++) printf(" %d", buf[i]);
    printf("\n");
#endif
    if ((int)tgt_len > tgt_phlen) {
       debug0(" tgt too long -> ignore\n");
       nbi++; ok=false; continue; // ignore: too many target words
    }
    else {
        // copy target phrase into output vector
      for (i=0; i<tgt_len; i++) target_vect[i] = (REAL) buf[i]; 	// careful: we cast int to float
      for (   ; i<tgt_phlen; i++) target_vect[i] = (REAL) oempty;
    }
  
      // decide wether the current phrase pair is valid in function of the flags
    if (!ok) {
      debug0(" -> late ignore\n");
      continue;
    }

    if (mode & DATA_PHRASE_IGN_SHORT_SRC) {
        // ignore phrase that don't have a source length identical to the input dimension
      if (src_len != src_phlen) {
        nbi++; ok=false; continue;
      }
    }

    if (mode & DATA_PHRASE_IGN_SHORT_TGT) {
        // ignore phrase that don't have a target length identical to the output dimension
      if (tgt_len != tgt_phlen) {
        nbi++; ok=false; continue;
      }
    }

    ok=true;
  } // of while (!ok)

  // read auxiliary data
  if (aux_fs.is_open()) {
    for (int i = 0; i < auxdim ; i++) {
      aux_fs >> input[idim + i];
      if (!aux_fs)
        Error("Not enough auxiliary data available");
    }
  }
  
#ifdef DEBUG
  printf("EX:");
  for (int i=0; i<idim; i++) printf(" %d", (int) input[i]); printf(" ->");
  for (int i=0; i<odim; i++) printf(" %d", (int) target_vect[i]); printf("\n");
#endif

  idx++;
  return true;
}

/********************
 *
 ********************/


void DataPhraseBin::Rewind()
{
  debug0("*** DataPhraseBin::Rewind()\n");
  lseek(fd, sizeof(int), SEEK_SET);	// position on field max_phrase_len
  int mlen;
  read(fd, &mlen, sizeof(int));		// get max_phrase_len
  uint pos = 2 * (sizeof(uint)+3*sizeof(WordID) + mlen*sizeof(int) );
  lseek(fd, pos , SEEK_CUR);
  if (aux_fs.is_open())
    aux_fs.seekg(0, aux_fs.beg);
  debug2("DataPhraseBin::Rewind(): max_phase_len=%d, advance by %u bytes\n", mlen, pos);
  idx=-1;
    // initialize read buffer
  buf_n=0; buf_pos=-1;
  debug0("*** DataPhraseBin::Rewind() done\n");
}
  
