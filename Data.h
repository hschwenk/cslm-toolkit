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

#ifndef _Data_h
#define _Data_h

#include <iostream>
#include <fstream>
#include <pthread.h>
#include <vector>
#include "Tools.h"
#include "DataFile.h"
#include "WordList.h"

// Names of information in files

#define DATA_LINE_LEN 16384   // extern const int gives internal gcc error for 4.7.2
extern const char* DATA_HEADER_TXT;
extern const int   DATA_HEADER_ID;
extern const char* DATA_PRELOAD;
extern const char* DATA_RESAMPL_MODE;
extern const char* DATA_RESAMPL_SEED;
extern const char* DATA_SHUFFLE_MODE;
extern const char* DATA_PATH_PREFIX;
extern const char* DATA_NB_FACTORS;
extern const char* DATA_SENTENCE_SCORES;
extern const char* DATA_AUXILIARY_DATA;
extern const char* DATA_WORD_LIST;
extern const char* DATA_WORD_LIST_TARGET;
extern const char* DATA_WORD_LIST_SOURCE;
extern const char* DATA_WORD_LIST_MULT;
extern const char* DATA_WITH_EOS;

#define DATA_FILE_BUF_SIZE 16384	// read large chunks form file for faster processing (used by some classes)


/*
 * Strategy
 *  - there is one function Rewind() and Next() which should not be overridden
 *  - they perform all the processing with preloading, shuffling, etc
 *  - the class specific processing is done in First() and Advance()
 */

typedef vector< vector<DataFile*> > FactoredDataFiles;

class Data
{
private:
  void CreateWordList(ifstream &ifs, size_t idx, bool use_class);
  void CompareWordList(size_t ii, Data &other_data, size_t ei);
  static void DeleteWordList(vector<WordList> *&iw, int *&is, pthread_mutex_t *&im);
  bool use_unstable_sort;	// switch for compatibility with CSLM <= V3.0 to use unstable sort
protected:
  const char *fname;
  char *path_prefix;	// prefix added to all file names
  int  idim, odim;	// dimensions
  int auxdim;		// auxiliary data dimension
  int nb_SentSc;        // Number of Sentence scores
  int ExpGrowth;        // V Value to be used for exponentielle growth for exp(-V/#Ep_nb)
  int Epoch_num;            // epoch number
  int tgpos;		// position of target
  int  nb_totl;		// number of examples
    // flags
  int preload;		// 
  int betweenSent_ctxt; // To be activated for continuous ngram between consecutive sentences 
  int resampl_mode;	// 
  int resampl_seed;	// 
  int shuffle_mode;	// 
  int norm_mode;	// evtl. perform normalization; bits: 1=subtract mean, 2=divide by var.
  int nb_factors;	// 

    // word lists
  vector<vector<WordList>*> wlist;
  vector<int*> w_shared;  		// number of objects sharing word list
  vector<pthread_mutex_t*> w_mutex;	// mutex used to share word list
  size_t sr_wl_idx, tg_wl_idx;	// source and target word list index
    // data files
  FactoredDataFiles  	datafile;
  vector<vector<pair<int, REAL*> > > df_dim;	// data file buffer (dim and ptr) for each word list
  int current_df;
    // actual data
  int  idx;		// index of current example [0,nb-1]
  int  *mem_cdf;	// current data file for each example in memory
  REAL *mem_inp;	// all the input data in memory
  REAL *mem_trg;	// all the output data in memory
    // constructor to create a void data object
  Data();
    // method to read content of data file
  virtual void ReadFile(Data *other_data = NULL, bool use_class = false);
    // local tools, only used when preload is activated
  void Preload();	// preload all data
  void Shuffle();	// shuffle in memory
public:
  Data(const char *fname, Data *other_data = NULL, bool use_class = false);
  virtual ~Data();
    // access function to local variables
  const char *GetFname() {return fname;}
  int GetIdim() {return idim;}
  int GetOdim() {return odim;}
  int GetNbFactors() {return nb_factors;}
  int GetNbSentSc() const { return nb_SentSc; }
  int GetAuxdim() const { return auxdim; }
  int GetTgPos() const { return tgpos; }
  int GetNb() {return nb_totl;}
  int GetIdx() {if (idx<0) Error("DataNext() must be called before GetIdx()"); return idx;};
  vector<WordList> *GetSrcWList() {return ((sr_wl_idx!=(size_t)-1) ? wlist[sr_wl_idx] : NULL);}
  vector<WordList> *GetTgtWList() {return ((tg_wl_idx!=(size_t)-1) ? wlist[tg_wl_idx] : NULL);}
    // the following two pointers are only valid after first DataNext() !
  REAL *input;		// pointer to current inputs
  REAL *target;		// pointer to current target
  REAL *aux;		// pointer to current auxiliary data
  //REAL *GetData() {return val;}
    // main functions to access data
  virtual int GetDim(size_t lg) const { return (((0 <= current_df) && (df_dim[current_df].size() > lg)) ? df_dim[current_df][lg].first : 0); }
  virtual REAL * GetBuffer(size_t lg) { return (((0 <= current_df) && (df_dim[current_df].size() > lg)) ? df_dim[current_df][lg].second : NULL); }
  virtual vector<WordList> *GetWList(size_t lg) {return ((wlist.size() > lg) ? wlist[lg] : NULL);}
  virtual void Rewind();	// rewind to first example, performs resampling, shuffling, etc if activated
  virtual bool Next();		// advance to next example, return FALSE if at end
};

#endif
