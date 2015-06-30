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
 * text2bin.cpp: tool to convert UTF8 texts to binary representation for the
 * CSLM toolkit
 *
 * The previous program convertToInt sorts the word list using native byte
 * values for the order. This can be reproduced by UNIX sort by setting
 * LC_ALL=C
 */

using namespace std;
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iostream>
#include "Tools.h"
#include "WordList.h"
#include "DataNgramBin.h"

#define LINE_LEN 1048576

int main (int argc, char *argv[]) {
  WordList *voc = new WordList(true);
  WordList *oov_w = new WordList;
  WordList::WordIndex nvoc, idx;
  char	line[LINE_LEN];
  bool do_lc=false;  // convert all input text to lowercase (we do not modify the vocabulary !)

  cout << "Text to binary converter " << cslm_version << ", H. Schwenk, Copyright 2014, LIUM, University of Le Mans, France" << endl;
    // parse args
  if (argc<5 || argc>6) {
    cerr << " usage: " << argv[0] << " input-vocab output-binary-file output-word-freq output-list-of-oov [--lc]< file" << endl;
    cerr << "        --lc     convert all input text to lowercase (the vocabulary is not modified !)" << endl;
    return 1;
  }
  char *voc_fname=argv[1];
  char *bin_fname=argv[2];
  char *wfreq_fname=argv[3];
  char *oov_fname=argv[4];
  if (argc==6) {
    if (strcmp(argv[5], "--lc")==0) {
      do_lc=true;
      cout << " - all input text is converted to lower case" << endl;
    }
    else ErrorN("unknown argument %s\n", argv[5]);
  }

    // read vocabulary
  voc->Read(voc_fname);
  nvoc=voc->GetSize();	// WordList may add <unk>, <s> or </s> to the provided word list !!
  WordList::WordIndex idx_unk=voc->GetIndex(WordList::WordUnknown);
  WordList::WordIndex idx_bos=voc->GetIndex(WordList::WordSentStart);
  WordList::WordIndex idx_eos=voc->GetIndex(WordList::WordSentEnd);
  printf(" - using word list %s (%d words, unk=%d, bos=%d, eos=%d)\n", voc_fname, nvoc, idx_unk, idx_bos, idx_eos);

    // write empty header (actual counts will be written at the end)
  int id=DATA_FILE_NGRAMBIN_VERSION;
  cout << " - writing binary representation V" << -id << " to file " << bin_fname << endl;
  FILE *binf = fopen(bin_fname, "wb");
  if (binf == NULL) {
    perror(bin_fname);
    Error();
  }
  
  ulong nl=0, nw=0, nunk=0;
  fwrite(&id, sizeof(id), 1, binf);	// header
  fwrite(&nl, sizeof(ulong), 1, binf);	// nb of lines
  fwrite(&nw, sizeof(ulong), 1, binf);	// nb of words
  fwrite(&nvoc, sizeof(nvoc), 1, binf);	// vocab size
  int i=sizeof(WordList::WordIndex);
  fwrite(&i, sizeof(int), 1, binf); // TODO
  fwrite(&idx_bos, sizeof(WordList::WordIndex), 1, binf);
  fwrite(&idx_eos, sizeof(WordList::WordIndex), 1, binf);
  fwrite(&idx_unk, sizeof(WordList::WordIndex), 1, binf);
  
    // read file, convert to binary, count word frequencies and #unk
  while (cin.getline(line, LINE_LEN)) {
    line[strlen(line)]=0;
    line[strlen(line)+1]=0;
    nl++;
    fwrite(&idx_bos, sizeof(idx_bos), 1, binf);
    
    char *bptr, *eptr;

    if (do_lc) {  // convert line to lower case
      for (bptr=line; *bptr; bptr++) *bptr=tolower(*bptr);
    }

    bptr=line;
    while ((*bptr != 0) && (*bptr != '\n') && (*bptr == ' ')) bptr++; /* skip blank */
    if (*bptr == '\n') continue;    /* skip empty lines */

      // loop on all words in line
    //cerr << "Line: " << line << endl;
    while ((*bptr != 0) && (*bptr != '\n')) {

      eptr = bptr + 1;
      while ((*eptr != 0) && (*eptr != '\n') && (*eptr != ' ')) eptr++;
      *eptr = 0;

      idx = voc->GetIndex(bptr);
      //cerr << bptr << "[" << idx <<"]"<< endl;
      nw++;
      if (nw%1000000 == 0) cout << "\r - processing " << nw/1000000 << "M words";
      if (idx==WordList::BadIndex) {
        fwrite(&idx_unk, sizeof(idx_unk), 1, binf);
        nunk++;
        idx=oov_w->AddWord(bptr);
#ifdef COUNT_OOV
        if (idx<0) ErrorN("illegal OOV idx (%d) for word %s\n",idx, bptr);
#endif
      }
      else {
        fwrite(&idx, sizeof(idx), 1, binf);
        if (idx<0 || idx>nvoc) ErrorN("illegal word index (%d) for word %s\n",idx, bptr);
        voc->GetWordInfo(idx).n++;
      }

      bptr = eptr + 1;
      while ((*bptr != 0) && (*bptr != '\n') && (*bptr == ' ')) bptr++;

    }
    fwrite(&idx_eos, sizeof(idx_eos), 1, binf);
    for (uint i=0; i<3; i++) line[i]=0; // clear beginning of the buffer 
  }
  cout << "\r";

    // dump vocabulary with word frequencies to file
  cout << " - dumping word frequencies to file " << wfreq_fname << endl;
  WordList::WordIndex ndiff = voc->Write(wfreq_fname, 2);	// TODO: should be uint or long

    // dump list of OOVs to file
  {
    cout << " - dumping list of OOV to file " << oov_fname << endl;
#ifdef COUNT_OOV
    oov_w->Write(oov_fname, 2);
#else
    oov_w->Write(oov_fname, 1);
#endif
  }

    // write header with actual values: id, nb_lines, nb_words, nbvoc, bos, eos, unk
  rewind(binf);
  fwrite(&id, sizeof(id), 1, binf);	// header
  fwrite(&nl, sizeof(nl), 1, binf);
  fwrite(&nw, sizeof(nw), 1, binf);
  fwrite(&nvoc, sizeof(nvoc), 1, binf);
  i=sizeof(WordList::WordIndex);
  fwrite(&i, sizeof(int), 1, binf);
  fwrite(&idx_bos, sizeof(WordList::WordIndex), 1, binf);
  fwrite(&idx_eos, sizeof(WordList::WordIndex), 1, binf);
  fwrite(&idx_unk, sizeof(WordList::WordIndex), 1, binf);

    // print final stats
  printf(" - %lu lines with %lu words processed, %u uniq words (%5.2f%% of the vocabulary)\n",
	nl, nw, ndiff, 100.0*ndiff/nvoc);
  printf(" - %lu words were unknown (%5.2f%% of the text), %d new words\n", nunk, 100.0*nunk/nw, oov_w->GetSize());

  fclose(binf);
  delete voc;
  delete oov_w;

  return 0;
}
