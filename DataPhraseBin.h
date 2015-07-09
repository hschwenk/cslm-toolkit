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

#ifndef _DataPhraseBin_h
#define _DataPhraseBin_h

#include <iostream>
#include <fstream>

#include "Tools.h"
#include "Data.h"
#include "DataFile.h"

#define DATA_FILE_PHRASE_BUF_SIZE (DATA_FILE_BUF_SIZE*sizeof(WordID))	// to have the same size than for DataNgraBin

extern const char* DATA_FILE_PHRASEBIN;
typedef unsigned char uchar;

// Syntax of a line in data description:
// DataPhraseBin <file_name> <resampl_coeff> <src_phrase_len> <tgt_phrase_len> [flags]
//   1: skip too short source phrases
//  16: skip too short target phrases
// Phrase pairs for which the source or target part is too long are always skipped
// (there is not reasonable way to "back-off" to a shorter phrase pair
//
// format of binary file
// header: (17 int = 68 bytes)
// int 		sizeof(WordID)
// int 		max_phrase_len
// uint		voc_size		\  source
// WordID	unk, bos, eos WordIDs	/
// int* 	array of number of source phrases for each length 1..max_phrase_len
// uint		voc_size		\  target
// WordID	unk, bos, eos WordIDs	/
// int* 	array of number of targ phrases for each length 1..max_phrase_len

class DataPhraseBin : public DataFile
{
private:
  void do_constructor_work();
protected:
  int fd;			// UNIX style binary file
  int max_len;			// max length wof words in phrase, read from file
  int mode;			// TODO
  int src_phlen, tgt_phlen;	// filter: max length of source and target phrases
    // input
  int ivocsize;			// vocab size (including <s>, </s> and <unk>)
  WordList *iwlist;		// pointer to source word list
  WordID ibos,ieos,iunk;	// word id of BOS, EOS and UNK in source vocabulary
  WordID iempty;		// word id of empty phrase (used to simulate shorter ones)
				// set to EOS if present in vocabulary, NULL_WORD else
  int *inbphw;			// array[max_len+1] of nb of phrases per length
				// indices start at 1, indice 0 gives the total count
  int *icnbphw;			// same, but cumulated number
    // ouput
  int ovocsize;			// vocab size (including <s>, </s> and <unk>)
  WordList *owlist;		// pointer to source word list
  WordID obos,oeos,ounk;	// word id of BOS, EOS and UNK in target vocabulary
  WordID oempty;		// word id of empty phrase (used to simulate shorter ones)
  int *onbphw, *ocnbphw;
    // stats (in addition to nbex in mother class)
  int nbi;			// ignored phrases (too long source or target part)

    // read buffer for faster File IO, we read BYTES not WordID !!
  uchar buf_bytes[DATA_FILE_PHRASE_BUF_SIZE];
  int buf_n;		// actual size of data in buffer
  int buf_pos;		// current position
  bool ReadBuffered(uchar *data, size_t cnt) {
#if 0
    read(fd, data, cnt);
#else
    debug2("DataPhraseBin::ReadBuffered(%p,%lu)\n",data,cnt);
    for (size_t i=0; i<cnt; i++) {
      if (++buf_pos>=buf_n) {
          // read new block of data, we can get less than requested
        buf_n = read(fd, buf_bytes, DATA_FILE_PHRASE_BUF_SIZE);
        debug1(" -put %d bytes into buffer\n", buf_n);
        if (buf_n<=0) return false; // no data left
        buf_pos=0;
      }
      debug2(" - copy bytes from buf[%d] to target[%lu]\n", buf_pos,i);
      data[i]=buf_bytes[buf_pos];
    }
#endif
    return true;
  }
public:
  DataPhraseBin(char*, ifstream&, int, const string&, int, const string&, int, DataPhraseBin* =NULL);	// optional object to initialize when adding factors
  DataPhraseBin(char*, float =1.0, int =5, int =5, int =17);
  virtual void SetWordLists(WordList*, WordList*);
  virtual ~DataPhraseBin();
  virtual bool Next();
  virtual void Rewind();
};

#endif
