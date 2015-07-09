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
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "Tools.h"
#include "Data.h"
#include "DataFile.h"

DataFile::DataFile(char *p_path_prefix, ifstream &ifs, int p_aux_dim, const string& p_aux_ext, int p_nb_SentSc,const string& p_SentSc_ext,int p_betweenSentCtxt , DataFile *prev_df)
 : idim(0), odim(0), auxdim(p_aux_dim), nbex(0), resampl_coeff(1.0), path_prefix(p_path_prefix), fname(NULL),
   nb_SentSc(p_nb_SentSc), betweenSent_ctxt(p_betweenSentCtxt), SentSc_ext(p_SentSc_ext), 
   idx(-1), input(NULL), target_vect(NULL), aux(NULL), target_id(0)
{
  debug0("** constructor DataFile\n");
  char p_fname[DATA_LINE_LEN];

  ifs >> p_fname;
    // prev_df is usefull for factors, since we don't repeat the resampl_coef for each factor, same for aux data
  if (prev_df) {
    resampl_coeff    = prev_df->resampl_coeff;
    auxdim           = prev_df->auxdim;
    aux_fname        = prev_df->aux_fname;
    SentSc_fname     = prev_df->SentSc_fname;
    SentSc_ext       = prev_df->SentSc_ext;
    nb_SentSc        = prev_df->nb_SentSc;
    betweenSent_ctxt = prev_df->betweenSent_ctxt;
    if (0 < auxdim) {
      aux_fs.open(aux_fname.c_str(), ios::in);
      CHECK_FILE(aux_fs, aux_fname.c_str());
    }
  }
  else {
    ifs >> resampl_coeff;	// read resampl coeff
    SetAuxDataInfo(p_aux_dim, p_aux_ext, p_fname);
  }
  if (resampl_coeff<=0 || resampl_coeff>1)
    Error("resampl coefficient must be in (0,1]\n");
  fname=strdup(p_fname);

  // memory allocation of input and target_vect should be done in subclass
  // in function of the dimension and number of examples
}

DataFile::DataFile(char *p_path_prefix, char *p_fname, const float p_rcoeff)
 : idim(0), odim(0), auxdim(0), nbex(0), resampl_coeff(p_rcoeff), path_prefix(p_path_prefix), fname(NULL),
   idx(-1), input(NULL), target_vect(NULL), aux(NULL), target_id(0)
{
  debug0("** constructor DataFile with fname\n");
  if (NULL != p_fname)
    fname = strdup(p_fname);

  // memory allocation of input and target_vect should be done in subclass
  // in function of the dimension and number of examples
}

DataFile::~DataFile()
{
  debug0("** destructor DataFile\n");
  if (fname) free(fname);
  if (aux_fs.is_open())
    aux_fs.close();
  // memory deallocation of input and target_vect should be done in subclass

}

/**
 * set auxiliary data information
 * @param dim dimension
 * @param ext file name extension
 * @param fname file name (with other extension)
 */
void DataFile::SetAuxDataInfo(int dim, const string& ext, char* fname)
{
  // get dimension and file name
  auxdim = dim;
  if (NULL != path_prefix) {
    aux_fname = path_prefix;
    aux_fname += '/';
  }
  else
    aux_fname.clear();
  aux_fname += ((NULL != fname) ? fname : this->fname);
  size_t dotpos = aux_fname.find_last_of('.');
  if (string::npos != dotpos)
    aux_fname.replace(dotpos + 1, string::npos, ext);
  else {
    aux_fname += '.';
    aux_fname += ext;
  }

  // open auxiliary file
  if (aux_fs.is_open())
    aux_fs.close();
  if (0 < auxdim) {
    cout << " - opening auxiliary data file " << aux_fname << endl;
    aux_fs.open(aux_fname.c_str(), ios::in);
    CHECK_FILE(aux_fs, aux_fname.c_str());
  }
}
/*******************
 *  return the resampling score. Return the avg if many
 * ******************/
double DataFile::GetResamplScore()
{
   int i;
   double r_score=0.0;

   if( nb_SentSc == 1 ) return SentScores[0];

   for(i=0;i<nb_SentSc;i++)
            r_score +=SentScores[i];

  return (r_score/nb_SentSc );
}


/**************************
 *
 **************************/

int DataFile::Info(const char *txt)
{
  ulong nbr = GetNbresampl();
  printf("%s%s  %6.4f * %13lu = %13lu\n", txt, fname, resampl_coeff, nbex, nbr);
  return nbr;
}

/**************************
 *
 **************************/

void DataFile::Rewind()
{
  debug0("*** DataFile::Rewind()\n");
  Error("DataFile::Rewind() should be overriden");
}

//*************************
// read next data in File
// Return false if EOF

bool DataFile::Next()
{
  Error("DataFile::Next() should be overriden");
  return false;
}

//**************************
// generic resampling function using sequential file reads
// cycles sequentially through data until soemthing was found
// based on DataNext() which may be overriden by subclasses
// returns idx of current example

int DataFile::Resampl()
{
  //debug0("*** DataFile::Resampl()\n");
  bool ok=false;

  while (!ok) {
    if (!Next()) Rewind(); // TODO: deadlock if file empty
//cout << "Resampled: ";
//for (int i=0; i<idim; i++) cout << input[i] << " ";
    ok = (drand48() < resampl_coeff);
//cout << " ok=" << ok << endl;
  }

  //debug0("*** DataFile::Resampl() end\n");
  return idx;
}

