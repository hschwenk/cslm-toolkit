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

#ifndef _DataFile_h
#define _DataFile_h

#include <iostream>
#include <fstream>
#include <vector>
#include "Tools.h"
#include "WordList.h"

class DataFile {
protected:
  int	idim, odim, auxdim;
  ulong  nbex;
  double resampl_coeff;
    // internal handling of data
  char *path_prefix;	// prefix to be added for each data file which will be opened, points to text allocated in Data.h
  char *fname;
  string aux_fname;	// auxiliary data file name
  ifstream aux_fs;	// auxiliary data file stream
  string SentSc_fname;  // Sentences scores file name
  ifstream SentSc_fs;   // Sentences scores file stream
  int nb_SentSc;        // Number of scores to be parsed from scores file
  int betweenSent_ctxt; // between Sentences  context
  string SentSc_ext;    // Sentences scores file extension
  void SetAuxDataInfo(int, const string&, char* = NULL);
public:
    // current data
  ulong  idx;
  REAL *input;		// current input data (needs to be allocated in subclass, we don't know the dimension yet)
  REAL *target_vect;	//         output data
  REAL *aux;		//         auxiliary data
  double *SentScores;   //         Sentences Scores vector
  int  target_id;	//         index of output [0..odim)
   // functions
  DataFile(char *, ifstream &, int, const string&, int, const string&,int , DataFile* =NULL);	// optional object to initialize when adding factors
  DataFile(char *, char *, const float =1.0);
  virtual ~DataFile();
   // access function
  int GetIdim() { return (idim + auxdim); }
  int GetOdim() { return odim; }
  ulong GetNbex() { return nbex; }
  ulong GetNbresampl() { return ((1.0 <= resampl_coeff) ? nbex : (ulong) (nbex * resampl_coeff)); }
  int GetNB_SentScores() {return nb_SentSc;}	
  double GetResamplCoef() { return resampl_coeff; }
  double GetResamplScore(); // used if SentenceScores option activated
  virtual void SetWordLists(WordList*, WordList*) {};	// only used in DataPhraseBin
   // main interface
  virtual int Info(const char* = " - ");	// display line with info after loading the data
  virtual void Rewind();	// rewind to first element
  virtual bool Next();		// advance to next data
  virtual int Resampl();	// resample another data (this may skip some elements in the file)
};

#endif
