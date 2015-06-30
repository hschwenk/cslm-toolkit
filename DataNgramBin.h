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

#ifndef _DataNgramBin_h
#define _DataNgramBin_h

#include <iostream>
#include <fstream>
#include <unistd.h>

#include "Data.h"
#include "DataFile.h"

extern const char* DATA_FILE_NGRAMBIN;
// ID of binary formet, we use negative numbers so that we can detect old files which have no version ID
// (the first field is number of lines which should be positive)
// 		id	nw	nl     voc_size	id_byte	beos	eos	unk
// no id:	n/a	int	int	int
// id=-2:	int	ulong	ulong	int		int	int	int
#define DATA_FILE_NGRAMBIN_VERSION2 (-2)	// introduced on Dec 02 2013
#define DATA_FILE_NGRAMBIN_VERSION DATA_FILE_NGRAMBIN_VERSION2
#define DATA_FILE_NGRAMBIN_HEADER_SIZE1 (2*sizeof(int)+sizeof(int)+sizeof(int)+3*sizeof(WordID))
#define DATA_FILE_NGRAMBIN_HEADER_SIZE2 (sizeof(int)+2*sizeof(ulong)+sizeof(int)+sizeof(int)+3*sizeof(WordID))

// Syntax of a line in data description:
// DataNgramBin <file_name> <resampl_coeff> <order> [<tgpos>] [flags]
//  u: skip n-grams with <unk> at the right most position
//  U: skip n-grams with <unk> anywhere
//  b: skip n-grams with <s> elsewhere than at the left most position
//  e: skip n-grams with </s> elsewhere than at the right most position

class DataNgramBin : public DataFile
{
private:
  void do_constructor_work();
  int id;		// ID to support different formats (see DATA_FILE_NGRAMBIN_VERSION)
  int header_len;	// length of header (in function of file version)
    // read buffer for faster File IO
  WordID buf_wid[DATA_FILE_BUF_SIZE];
  int buf_n;		// actual size of data in buffer
  int buf_pos;		// current position
  bool ReadBuffered(WordID *wid) {
    if (++buf_pos>=buf_n) {
        // read new block of data, we can get less than requested
      buf_n = read(fd, buf_wid, DATA_FILE_BUF_SIZE*sizeof(WordID)) / sizeof(WordID);
//printf("put %d elements into buffer\n", buf_n);
      if (buf_n<=0) return false; // no data left
      buf_pos=0;
    }
    *wid=buf_wid[buf_pos];
    return true;
  }
protected:
  int fd;		// UNIX style binary file
  int vocsize;		// vocab size (including <s>, </s> and <unk>)
  int order;		// order of the ngrams
  int tgpos;		// position of target word in n-gram
  int eospos;		// position of eos word in n-gram
  int mode;		// see above for possible flags
  WordID *wid;		// whole n-gram context
  WordID bos, eos, unk;	// word ids of special symbols
    // stats (in addition to nbex in mother class)
  ulong  nbl, nbw, nbs, nbu;// lines, words, sentences, unks
  ulong  nbi;		// ignored n-grams
public:
  explicit DataNgramBin(char*, ifstream&, int, const string&, int, const string&, int, DataNgramBin* =NULL);	// optional object to initialize when adding factors
  DataNgramBin(char*, float =1.0, int =4);
  DataNgramBin(char*, float, int, int, int =3);
  virtual ~DataNgramBin();
  virtual bool Next();
  virtual void Rewind();
  virtual WordID GetVocSize() {return vocsize;};
  int GetTgPos() const { return tgpos; }
};

#endif
