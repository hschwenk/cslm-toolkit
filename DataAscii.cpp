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

#include "Tools.h"
#include "Data.h"
#include "DataAscii.h"

const char* DATA_FILE_ASCII="DataAscii";

DataAscii::DataAscii(char *p_prefix, ifstream &ifs, int p_aux_dim, const string& p_aux_ext, int p_nb_SentSc, const string& p_SentSc_ext,int p_betweenSentCtxt , DataAscii *prev_df)
 : DataFile::DataFile(p_prefix, ifs, p_aux_dim, p_aux_ext, p_nb_SentSc, p_SentSc_ext, p_betweenSentCtxt, prev_df)
{

  char full_fname[max_word_len]="";

  if (path_prefix) {
    if (strlen(path_prefix)+strlen(fname)+2>(size_t)max_word_len)
      Error("full filename is too long");

    strcpy(full_fname, path_prefix);
    strcat(full_fname, "/");
  }
  strcat(full_fname, fname);

  dfs.open(full_fname,ios::in);
  CHECK_FILE(dfs,full_fname);

  if (prev_df) {
    nbex=prev_df->nbex;
    idim=prev_df->idim;
    odim=prev_df->odim;
    printf(" - %s: ASCII data with %lu examples of dimension %d -> %d (factor)\n", fname, nbex, idim, odim);
  }
  else {
    char buf[DATA_LINE_LEN];
    dfs.getline(buf,DATA_LINE_LEN);
    sscanf(buf, "%lu %d %d", &nbex, &idim, &odim);
    printf(" - %s: ASCII data with %lu examples of dimension %d -> %d\n", fname, nbex, idim, odim);
  }

  if (idim>0) input = new REAL[idim + auxdim];
  if (odim>0) target_vect = new REAL[odim];
}


/**************************
 *  
 **************************/

DataAscii::~DataAscii()
{
  dfs.close();
  if (idim>0) delete [] input;
  if (odim>0) delete [] target_vect;
}


/**************************
 *  
 **************************/

void DataAscii::Rewind()
{
  dfs.seekg(0, dfs.beg);
  char buf[DATA_LINE_LEN];
  dfs.getline(buf,DATA_LINE_LEN);
  if (aux_fs.is_open())
    aux_fs.seekg(0, aux_fs.beg);
}

/**************************
 *  
 **************************/

bool DataAscii::Next()
{
  char line[DATA_LINE_LEN];
  dfs.getline(line, DATA_LINE_LEN);
  if (dfs.eof()) return false;
          else idx++;

    // parse input data
  char *lptr=line;
//cout << "\nLINE: " << line << endl;
  for (int i=0; i<idim; i++) {
//cout << "parse:" <<lptr<<"; ";
    while (*lptr==' ' || *lptr=='\t') lptr++;
    if (!*lptr) {
      sprintf(line, "incomplete input in ASCII datafile, field %d", i);
      Error(line);
    }
    if (sscanf(lptr, "%f", input+i)!=1) Error("parsing source in ASCII datafile");
//cout << "got i[" <<i << "] " << input[i] << endl;
    while (*lptr!=' ' && *lptr!='\t' && *lptr!=0) lptr++;
  }

  // parse auxiliary data
  if (aux_fs.is_open()) {
    for (int i = 0; i < auxdim ; i++) {
      aux_fs >> input[idim + i];
      if (!aux_fs)
      {
        cout << " - Error in auxiliary data file: " << aux_fname << endl;
        Error("Not enough auxiliary data available");
      }
    }
  }

  if (odim<=0) return true;

    // parse target data
  for (int i=0; i<odim; i++) {
//cout << "parse:" <<lptr<<"; ";
    while (*lptr==' ' || *lptr=='\t') lptr++;
    if (!*lptr) {
      sprintf(line, "incomplete target in ASCII datafile, field %d", i);
      Error(line);
    }
    if (sscanf(lptr, "%f", target_vect+i)!=1) Error("parsing target in ASCII datafile");
//cout << "got t[" <<i << "] " << target_vect[i] << endl;
    while (*lptr!=' ' && *lptr!='\t' && *lptr!=0) lptr++;
  }

  return true;
}

