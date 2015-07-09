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
 */

using namespace std;
#include <iostream>
#include <cstring>
#include "BackoffLmSri.h"

//
//
//

void BackoffLmSri::BackoffLmSri_init(char *p_fname, int p_max_order)
{
  if ((p_fname == NULL) || (p_fname[0] == '\0')) {
    // no back-off file
    sri_vocab = NULL;
    sri_ngram = NULL;
    sri_order = p_max_order;
    sri_context_idxs = NULL;
    return;
  }

  if (p_max_order < 2)
    Error ("unsupported order of the SRI LM"); // TODO: give the actual order

  sri_vocab = new Vocab();
 
  if (strstr(p_fname,".vocab")) {
    cout << " - vocabulary " << p_fname << "was specified instead of an LM" << endl;

    sri_vocab->unkIsWord() = true;
    sri_vocab->toLower() = false;
    {
      File file(p_fname, "r");
      sri_vocab->read(file);
      //voc->remove("-pau-");
    }
    cout << "  found "<< sri_vocab->numWords() << ", returning lnProp=" << NULL_LN_PROB << "in all calls" << endl;
    
    sri_order=p_max_order; // TODO: is this correct
    sri_ngram = NULL;
  }
  else {
    cout << " - reading back-off SRILM from file '" << p_fname << "'" << endl;
    sri_ngram = new Ngram(*sri_vocab, p_max_order);

      // reading SRI LM
    sri_ngram->setorder(p_max_order);
    sri_ngram->skipOOVs() = false;
    File ngram_file(p_fname, "r");
    sri_ngram->read(ngram_file, 0);

      // get number of n-grams
      // TODO: can we get the order of the model read from file ?
    vector<uint> nb_ngrams;
    nb_ngrams.push_back(sri_vocab->numWords());
    cout << "   vocabulary: " << nb_ngrams[0] << " words; ngrams:";
    sri_order=0;
    for (int o=1; o<=p_max_order; o++) {
      nb_ngrams.push_back(sri_ngram->numNgrams(o));
      cout << " " << nb_ngrams.back();
      if (nb_ngrams.back()>0) sri_order++;
    }
  }

  cout << " (order=" << sri_order << ")" << endl;
  if (sri_order > p_max_order) {
    cout << " - limiting order of the back-off LM to the order of the CSLM (" << p_max_order << ")" << endl;
     sri_order = p_max_order;
  }

#ifdef LM_SRI0
  for (i=wlist.begin(); i!=wlist.end(); i++) {
    int sri_idx = sri_vocab->getIndex((*i).word);
printf("word=%s, sri=%d, wlist=%d\n", (*i).word, sri_idx, (*i).id);
  }
#endif

    // reserve memory for the context in SRI format
  sri_context_idxs = new VocabIndex[sri_order+1];
  sri_context_idxs[sri_order-1]=Vocab_None; // terminate, this is needed to specify the length of the context

  map_cslm2sri.clear();
}

//
//
//

BackoffLmSri::BackoffLmSri(char *p_fname, int p_max_order, const WordList &wlist)
{
  BackoffLmSri::BackoffLmSri_init(p_fname, p_max_order);
  if (NULL == sri_vocab)
    return;

    // set up mapping from CSLM indices to SRI LM indices
  cout << " - setting up mapping from CSLM to SRI word list" << endl;
  WordList::WordIndex wlsz = wlist.GetSize();
  map_cslm2sri.reserve(wlsz);
  map_cslm2sri.resize(wlsz);
  WordList::const_iterator iter = wlist.Begin(), end = wlist.End();
  for (size_t ci=0; iter!=end; iter++, ci++) {
    VocabIndex vi = sri_vocab->getIndex(iter->word);
    //debug3("'%s' bin=%d -> sri=%d\n", iter->word, ci, vi);
    if (vi == Vocab_None) {
      fprintf(stderr,"word %s not found at pos %zu\n", iter->word, ci );
    }
    else
      map_cslm2sri[ci] = vi;
  }
}

BackoffLmSri::~BackoffLmSri() {
  if (sri_vocab) delete sri_vocab;
  if (sri_ngram) delete sri_ngram;     
  map_cslm2sri.clear();
 if (sri_context_idxs)  delete [] sri_context_idxs;
}

/**
 * gets WordID of words in sentence
 * @param wid output table of WordID (allocated internally)
 * @param sentence input sentence
 * @param bos start sentence with BOS
 * @param eos end sentence with EOS
 * @return number of words
 */
int BackoffLmSri::GetSentenceIds(WordID *&wid, const string &sentence, bool bos, bool eos)
{
  if (NULL == sri_vocab)
    return 0;

  int nw = 0;
  static char str[max_words*16];
  static VocabString vstr[max_words-1];

  strcpy(str,sentence.c_str()); // we need to copy since parseWords() modifies the string
  nw = sri_vocab->parseWords(str, vstr, max_words - 1);
  if (nw >= max_words-1) Error("too many words in one hypothesis\n");
  debug1(" parsing found %d words\n", nw);

  int b=0;
    // start sentence with BOS ?
  if (bos) wid_table[b++]=sri_vocab->ssIndex();

  sri_vocab->getIndices(vstr, (VocabIndex*) (wid_table+b), nw + 1, sri_vocab->unkIndex());
#ifdef DEBUG
  for (int i=0;i<nw; i++) printf(" %s[%d]", vstr[i], wid_table[i+b]); cout<<endl;
#endif

    // end sentence with EOS ?
  nw += b;
  if (eos) wid_table[nw++]=sri_vocab->seIndex();

  wid = wid_table;
  debug4("* split sent with %d words into %d-grams (bos=%d, eos=%d):\n", nw, sri_order, sri_vocab->ssIndex(), sri_vocab->seIndex());
  return nw;
}
