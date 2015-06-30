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
#include "DataAsciiClass.h"

const char* DATA_FILE_ASCIICLASS="DataAsciiClass";

DataAsciiClass::DataAsciiClass(char *p_prefix, ifstream &ifs, int p_aux_dim, const string& p_aux_ext, int p_nb_SentSc, const string& p_SentSc_ext,int p_betweenSentCtxt, DataAsciiClass *prev_df)
 : DataAscii::DataAscii(p_prefix, ifs, p_aux_dim, p_aux_ext, p_nb_SentSc, p_SentSc_ext, p_betweenSentCtxt, prev_df)
{

  if (prev_df) {
    tgt0=prev_df->tgt0;
    tgt1=prev_df->tgt1;
    printf("   targets %5.2f/%5.2f (factor)\n", tgt0, tgt1);
  }
  else {
    ifs >> tgt0 >> tgt1;
    printf("   targets %5.2f/%5.2f\n", tgt0, tgt1);
  }
}


/**************************
 *  
 **************************/

bool DataAsciiClass::Next()
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
        Error("Not enough auxiliary data available");
    }
  }

  if (odim<=0) return true;

    // parse target data
  while (*lptr==' ' || *lptr=='\t') lptr++;
  if (!*lptr) Error("unable to parse target id in ASCII datafile");
  if (sscanf(lptr, "%d", &target_id)!=1) Error("parsing target in ASCII datafile");
  for (int t=0; t<odim; t++) target_vect[t]=tgt0;
  target_vect[target_id]=tgt1;

  return true;
}

