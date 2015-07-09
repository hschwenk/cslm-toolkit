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
 *
 *
 * This is a tool to convert extract files created by the Moses scoring tools
 * into the binary representation of the CSLM toolkit
 *
 */

using namespace std;
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include "Tools.h"
#include "Hypo.h"	// for NBEST_DELIM
#include "WordList.h"


const int LINE_LEN=65536;
#define MAX_PHRASE_LEN 32
typedef unsigned char uchar;

class Phrase
{
  private:
    char *msg;		// identifying message for display
    WordList *voc;	// vocabulary
    WordList::WordIndex nvoc;	// size of vocabulary
    bool create_voc;	// shall we create a new vocabulary or use a specified one ?
    WordList::WordIndex idx_unk, idx_bos, idx_eos;
    int max_len;	// maximum length of phrases (# of words)
    uint nw;		// total number of words in all phrases
    uint nwunk;		// total number of unknown words in all phrases
    uint np;		// total number of seen phrases 
    uint np_ok;		// total number of kept phrases 
    uint np_unk;	// number of phrases with at least one unknown word
    uint np_skip;	// number of phrases skipped since too long
    vector<uint> nphw;	// number of phrases in function of the number of words in the phrase
  public:
    Phrase(char *, char *, int=MAX_PHRASE_LEN);
    ~Phrase();
    int AddPhrase(FILE*, char*);
    void Info();
    void WriteHeader(FILE*);
    void WriteWlist(char*);
    void WriteUnk(char*) {};
};


//*********************************************************************

Phrase::Phrase(char *fname, char *p_msg, int p_max_len) 
  : nvoc(0), create_voc(true), max_len(p_max_len), nw(0), nwunk(0), np(0), np_ok(0), np_unk(0), np_skip(0) 
{
  for (int i=0; i<=max_len; i++) nphw.push_back(0);

  msg=strdup(p_msg);
  voc = new WordList(true);

  if (strcmp(fname,"-")==0) {
    printf(" - creating new %s word list (", msg);
    voc->AddWord(WordList::WordUnknown);
    voc->AddWord(WordList::WordSentStart);
    voc->AddWord(WordList::WordSentEnd);
    nvoc = voc->GetSize();	// WordList may add <unk>, <s> or </s> to the provided word list !!
  }
  else {
    voc->Read(fname);
    nvoc = voc->GetSize();	// WordList may add <unk>, <s> or </s> to the provided word list !!
    printf(" - using %s word list %s (%d words, ", msg, fname, nvoc);
    create_voc=false;
  }
  idx_unk = voc->GetIndex(WordList::WordUnknown);
  idx_bos = voc->GetIndex(WordList::WordSentStart);
  idx_eos = voc->GetIndex(WordList::WordSentEnd);
  printf("unk=%d, bos=%d, eos=%d)\n", idx_unk, idx_bos, idx_eos);
}

//*********************************************************************

Phrase::~Phrase() 
{
  free(msg);
  delete voc;
}

//*********************************************************************

int Phrase::AddPhrase(FILE *binf, char *line)
{
  char	*bptr = line, *eptr, *sptr;
  bool	found_unk=false;

  while ((*bptr != 0) && (*bptr != '\n') && (*bptr == ' ')) bptr++; /* skip initial blank */
  if (*bptr == '\n') return 0;    /* skip empty lines */
  sptr = bptr; // memorize
  np++;
  //  if (np%1000000 == 0) cout << "\r - processing " << np/1000000 << "M phrases";

    // count number number of words and write it into the file
  int	nw_in_phr=0;
  while ((*bptr != 0) && (*bptr != '\n')) {
    eptr = bptr + 1;
    while ((*eptr != 0) && (*eptr != '\n') && (*eptr != ' ')) eptr++;
    *eptr = 0;
    nw_in_phr++;

    bptr = eptr + 1;
    while ((*bptr != 0) && (*bptr != '\n') && (*bptr == ' ')) bptr++;
  }
  if (nw_in_phr > 255) {
    fclose(binf);
    Error("the binary format does not support more than 255 words per phrase");
  }
  uchar x= (uchar) nw_in_phr;
  fwrite(&x, sizeof(uchar), 1, binf);	// 1 byte is enough
  debug2("%s dump %d words:", msg, x);
    // loop on all words in line
  bptr=sptr;
  while ((*bptr != 0) && (*bptr != '\n')) {

    eptr = bptr + 1;
    while ((*eptr != 0) && (*eptr != '\n') && (*eptr != ' ')) eptr++;
    *eptr = 0;

    WordList::WordIndex idx = voc->GetIndex(bptr);
    nw++;
    if (idx==WordList::BadIndex) {
      if (create_voc) {
	nvoc++;
        idx=voc->AddWord(bptr);
        fwrite(&idx, sizeof(WordList::WordIndex), 1, binf);
      }
      else {
        nwunk++;
        found_unk=true;
        voc->GetWordInfo(idx_unk).n++;
        fwrite(&idx_unk, sizeof(WordList::WordIndex), 1, binf);
      }
      debug2(" UNK: %s[%d]", bptr,idx);
      //idx=unk_w->AddWord(bptr); TODO
#ifdef COUNT_OOV
      if (idx<0) ErrorN("illegal OOV idx (%d) for word %s\n",idx, bptr);
      if (idx>unk_cnt.capacity()) unk_cnt.reserve(2*unk_cnt.capacity());
      unk_cnt[idx]++;  // TODO: resize vector ??
#endif
    }
    else {
      if (idx<1 || idx>nvoc) ErrorN("illegal word index (%d) for %s word %s\n", idx, msg, bptr);
      voc->GetWordInfo(idx).n++;
      fwrite(&idx, sizeof(WordList::WordIndex), 1, binf);
      debug2(" %s[%d]", bptr,idx);
    }

    bptr = eptr + 1;
    while ((*bptr != 0) && (*bptr != '\n') && (*bptr == ' ')) bptr++;

  }
  // TODO for (i=0; i<LINE_LEN; i++) line[i]=0; // TODO: we need to clear the buffer !?
  debug0("\n");

    // stats
  nphw[nw_in_phr]++;
  if (nw_in_phr>max_len) np_skip++;
  if (found_unk) np_unk++;
  if (nw_in_phr<=max_len && !found_unk) np_ok++;

  return nw_in_phr;
}

//*********************************************************************

void Phrase::Info() 
{
  printf(" -   processed %d words, %d were unknown (%5.2f%%)", nw, nwunk, 100.0*nwunk/nw);
  if (create_voc) printf(", created new vocabulary with %d words\n", nvoc); else printf("\n");
  printf(" -   phrases: %d seen\n", np);
  if (!create_voc)
    printf(" -            %d (%5.2f%%) contained at least one unkown word\n", np_unk, 100.0*np_unk/np);
  printf(" -            %d (%5.2f%%) contained more than %d words\n", np_skip, 100.0*np_skip/np, max_len);
  printf(" -        =>  %d (%5.2f%%) phrases were kept\n", np_ok, 100.0*np_ok/np);
  printf(" -   phrase distribution per number of words in each phrase:\n");
  printf("     "); for (int i=1; i<=max_len; i++) printf("\t%9d", i);
  printf("\n     "); for (int i=1; i<=max_len; i++) printf("\t%9d", nphw[i]);
  printf("\n     "); for (int i=1; i<=max_len; i++) printf("\t%8.2f%%", 100.0*nphw[i]/np);
  printf("\n     "); for (int n=0,i=1; i<=max_len; i++) {n+=nphw[i]; printf("\t%8.2f%%", 100.0*n/np);} printf("  cumulated\n");
}

//*********************************************************************

void Phrase::WriteHeader(FILE *binf)
{
  fwrite(&nvoc, sizeof(uint), 1, binf);
  fwrite(&idx_unk, sizeof(WordList::WordIndex), 1, binf);
  fwrite(&idx_bos, sizeof(WordList::WordIndex), 1, binf);
  fwrite(&idx_eos, sizeof(WordList::WordIndex), 1, binf);
  for (int i=1; i<=MAX_PHRASE_LEN; i++) {
    int h=nphw[i];
    fwrite(&h, sizeof(int), 1, binf);	// number of phrases per length
  }
}

//*********************************************************************

void Phrase::WriteWlist(char *fname)
{
  cout << " - dumping " << msg << " word frequencies to file " << fname;
  WordList::WordIndex ndiff = voc->Write(fname, 2);
  cout << ", " << ndiff << " words had non zero frequency" << endl;
}

//*********************************************************************
// We store the number of source phrase per number of words
// By these means cstm_train doesn't need to count if limit on source and target words is identical

void WriteGlobalHeader(FILE *binf, Phrase &ph)
{
  int i;

  i=sizeof(WordList::WordIndex); fwrite(&i, sizeof(int), 1, binf);	// size of internal indices
  i=MAX_PHRASE_LEN;
  fwrite(&i, sizeof(int), 1, binf);			// max length of phrases
}

//*********************************************************************

int main (int argc, char *argv[]) {
  char	line[LINE_LEN];

  cout << "Phrase extraction to binary converter V1.0 2014, H. Schwenk, LIUM, University of Le Mans, France" << endl;
    // parse args
  if (argc!=8) {
    cerr << " usage: " << argv[0] << " output-binary-file   input-vocab input-word-freq input-list-of-unk   output-vocab output-word-freq output-list-of-unk < file" << endl;
    return 1;
  }
  char *bin_fname=argv[1];
  char *in_voc_fname=argv[2];
  char *in_wfreq_fname=argv[3];
  char *in_unk_fname=argv[4];
  char *out_voc_fname=argv[5];
  char *out_wfreq_fname=argv[6];
  char *out_unk_fname=argv[7];

  Phrase srcph(in_voc_fname, (char*) "source");
  Phrase tgtph(out_voc_fname, (char*) "target");

    // write empty header (actual counts will be written at the end)
  cout << " - writing binary representation to file " << bin_fname << endl;
  FILE *binf = fopen(bin_fname, "wb");
  if (binf == NULL) {
    perror(bin_fname);
    Error();
  }
  WriteGlobalHeader(binf, srcph);
  srcph.WriteHeader(binf);
  tgtph.WriteHeader(binf);
  
  int np=0; // number of phrase pairs = lines
  while (cin.getline(line, LINE_LEN)) {
    line[strlen(line)]=0;
    line[strlen(line)+1]=0;
    np++;
    
      // find source part
    char *bptr=line, *eptr;
    if ((eptr=strstr(bptr,NBEST_DELIM))==NULL) {
        fclose(binf);
        ErrorN("can't find the source phrase in line %d:\n%s", np, line);
    }
    *eptr=0;
    srcph.AddPhrase(binf, bptr);

      // find target part
    bptr=eptr+strlen(NBEST_DELIM);
    if ((eptr=strstr(bptr,NBEST_DELIM))==NULL) {
        fclose(binf);
        ErrorN("can't find the target phrase in line %d:\n%s", np, line);
    }
    *eptr=0;
    tgtph.AddPhrase(binf, bptr);

  }

    // write header with actual values: 
  rewind(binf);
  WriteGlobalHeader(binf, srcph);
  srcph.WriteHeader(binf);
  tgtph.WriteHeader(binf);
  fclose(binf);
    
    // print final stats
  printf(" - %d phrase pairs processed\n", np);
  cout << " - statistics on source part:" << endl;
  srcph.Info();
  cout << " - statistics on target part:" << endl;
  tgtph.Info();

  srcph.WriteWlist(in_wfreq_fname);
  srcph.WriteUnk(in_unk_fname);
  tgtph.WriteWlist(out_wfreq_fname);
  tgtph.WriteUnk(out_unk_fname);

  return 0;
}
