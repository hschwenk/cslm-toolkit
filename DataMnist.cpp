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
#include "Data.h"
#include "DataMnist.h"

const char* DATA_FILE_MNIST="DataMnist";
const uint magic_mnist_data=0x00000803;
const uint magic_mnist_labels=0x00000801;

/*
 * 
 */

uint DataMnist::read_iswap(int fd) {
  uint i, s;
  unsigned char *pi=(unsigned char*) &i, *ps=(unsigned char*) &s;

  read(fd, &i, sizeof(i));
  
  // swap integer Big Endian -> little Endian
  ps[0]=pi[3]; ps[1]=pi[2]; ps[2]=pi[1]; ps[3]=pi[0];
  debug2("read=%4x, swap=%4x\n", i, s);
   
  return s;
}

/*
 * 
 */

DataMnist::DataMnist(char *p_prefix, ifstream &ifs, int p_aux_dim, const string& p_aux_ext, int p_nb_SentSc, string& p_SentSc_ext,int p_betweenSentCtxt, DataMnist *prev_df)
 : DataFile::DataFile(p_prefix, ifs, p_aux_dim, p_aux_ext, p_nb_SentSc, p_SentSc_ext, p_betweenSentCtxt, prev_df)
{
  debug0("** constructor DataMnist\n");
  char full_fname[max_word_len]="";

  printf(" - %s: MNIST data ", fname); fflush(stdout);

   // open data file (fname is parsed by DataFile::DataFile()
  if (path_prefix) {
    if (strlen(path_prefix)+strlen(fname)+2>(size_t)max_word_len)
      Error("full filename is too long");

    strcpy(full_fname, path_prefix);
    strcat(full_fname, "/");
  }
  strcat(full_fname, fname);
  
  dfd=open(full_fname, O_RDONLY);
  if (dfd<0) perror("");
  if (read_iswap(dfd) != magic_mnist_data) Error("magic number of data file is wrong");
  nbex = read_iswap(dfd);
  idim = read_iswap(dfd) * read_iswap(dfd);
  printf("with %lu examples of dimension %d\n", nbex, idim);

   // open corresponding label file
  if (prev_df) {
    cl_fname=prev_df->cl_fname;
    odim=prev_df->odim;
    tgt0=prev_df->tgt0;
    tgt1=prev_df->tgt1;
    printf("   %s with labels in %d classes, targets %5.2f %5.2f (factor)\n", cl_fname, odim, tgt0, tgt1); fflush(stdout);
  }
  else {
    char p_clfname[DATA_LINE_LEN];
    ifs >> p_clfname >> odim >> tgt0 >> tgt1;
    cl_fname=strdup(p_clfname);
    printf("   %s with labels in %d classes, targets %5.2f %5.2f\n", cl_fname, odim, tgt0, tgt1); fflush(stdout);
  }

  full_fname[0]=0;
  if (path_prefix) {
    if (strlen(path_prefix)+strlen(cl_fname)+2>(size_t)max_word_len)
      Error("full filename is too long");

    strcpy(full_fname, path_prefix);
    strcat(full_fname, "/");
  }
  strcat(full_fname, cl_fname);

  lfd=open(full_fname, O_RDONLY);
  if (lfd<0) perror(""); 
  ulong val;
  if (read_iswap(lfd) != magic_mnist_labels) Error("magic number of label file is wrong");
  if ((val=read_iswap(lfd)) != nbex) ErrorN("found %lu examples in label file", val);
  
  if (idim>0) {
    input = new REAL[idim + auxdim];
    ubuf = new unsigned char[idim];
  }
  if (odim>0) target_vect = new REAL[odim];
}


/**************************
 *  
 **************************/

DataMnist::~DataMnist()
{
  debug0("** destructor DataMnist\n");
  close(dfd);
  close(lfd);
  if (idim>0) { delete [] input; delete [] ubuf; }
  if (odim>0) delete [] target_vect;
  if (cl_fname) free(cl_fname);
}


/**************************
 *  
 **************************/

void DataMnist::Rewind()
{
  debug0("*** DataMnist::Rewind()\n");
  lseek(dfd, 16, SEEK_SET);
  lseek(lfd, 8, SEEK_SET);
  if (aux_fs.is_open())
    aux_fs.seekg(0, aux_fs.beg);
}

/**************************
 *  
 **************************/

bool DataMnist::Next()
{
//  debug0("*** DataMnist::Next() "); cout<<idx<< " << endl;
 
    // read next image 
  int t=idim*sizeof(unsigned char);
  if (read(dfd, ubuf, t) != t) return false;

  for (t=0; t<idim; t++) input[t]= (REAL) ubuf[t];
#ifdef DEBUG
  int x,y;
  printf("\nEX %lu\n", idx);
  for (y=0; y<28; y++) {
    for (x=0; x<28; x++) printf("%d", ubuf[y*28+x] > 16 ? 1 : 0);
    printf("\n");
  }
#endif

  // read auxiliary data
  if (aux_fs.is_open()) {
    for (int i = 0; i < auxdim ; i++) {
      aux_fs >> input[idim + i];
      if (!aux_fs)
        Error("Not enough auxiliary data available");
    }
  }

    // read next class label
  if (odim<=0) return true;
  if (read(lfd, ubuf, 1) != 1) {
    char msg[16384];  // TODO
    sprintf(msg, "no examples left in class file %s", cl_fname);
    Error(msg);
  }
  debug1("class %d\n", ubuf[0]);
  target_id = (int) ubuf[0];
  if (target_id>=odim) {
    ErrorN("example %lu has a target of %d, but we have only %d classes\n", idx+1, target_id, odim);
  }
  for (t=0; t<odim; t++) target_vect[t]=tgt0;
  target_vect[target_id]=tgt1;

  idx++;
  return true;
}

