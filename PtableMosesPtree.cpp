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

#include "PtableMosesPtree.h"


PtableMosesPtree::~PtableMosesPtree ()
{
  for (vector<Moses::PhraseDictionaryTree*>::iterator p=ptree.begin(); p!=ptree.end(); p++)
    (*p)->FreeMemory();
}

//
// read a new phrase table
//
void PtableMosesPtree::Read(const string &fname, const int p_nscores, const char *scores_specif)
{
  if (strlen(scores_specif)<2 || scores_specif[1]!=':')
    Error("format error in the specification of the TM scores");
  if (scores_specif[0]<'1' || scores_specif[0]>'4')
    Error("wrong value for the number of TM scores");

  if (ptree.size()==0)
    nscores=scores_specif[0]-'0';
  else {
    if (nscores!=scores_specif[0]-'0')
      Error("PtableMosesPtree::Read(): inconsistent number of scores to be returned from multiple phrase tables");
  }
  if (nscores > p_nscores)
    Error("PtableMosesPtree::Read(): the number of scores to be returned exceeds the number of available ones");

  ptree.push_back(new Moses::PhraseDictionaryTree);
  pos_scores.push_back(scores_specif[2]-'0');

  ptree.back()->NeedAlignmentInfo(false);
  cout << " - loading Moses binary phrase table from file " << fname << " with " << p_nscores << " scores" << endl;
  ptree.back()->Read(fname);
  cout << "   using " << nscores << " scores starting at position " << pos_scores.back() << endl;
  tgtcands.clear();
};


//
// Get probabilities from the phrase-tables
//  - scores=NULL:	return either one value as a function result
//  - scores!=NULL:	return a sequence of values in that vector (as many as the vector has space)
//

REAL PtableMosesPtree::GetProb(vector<string> &src, vector<string> &tgt, vector<float> *scores)
{
  uint w;

#ifdef DEBUG
  cout << "Ptable prob:";
  for (w=0; w<src.size(); w++) cout << " " << src[w];
  cout << " |||";
  for (w=0; w<tgt.size(); w++) cout << " " << tgt[w];
  cout << " ||| " << endl;
#endif

  if (scores && scores->size() == 0)
    Error("PtableMosesPtree::GetProb() parameter scores has zero dimension");

  if (scores && (int) scores->size() > nscores)
    Error("PtableMosesPtree::GetProb() requesting too much scores form the phrase table");


  for (uint p=0; p<ptree.size(); p++) {

      // get all target phrases with scores from current phrase table
    tgtcands.clear();
    ptree[p]->GetTargetCandidates(src, tgtcands);
    debug2(" - phrase table %u has %d candidates:\n", p, (int) tgtcands.size());
    size_t pos=pos_scores[p];

      // search for our target phrase
    for (uint tph=0; tph<tgtcands.size(); tph++) {
      //debug2(" - candidate %d, length %d\n", tph, (int) tgtcands[tph].tokens.size());
      if (tgt.size() != tgtcands[tph].tokens.size()) continue;
      bool match=true;
      for (w=0; match && w<tgt.size(); w++) {
        match = (tgt[w].compare(*(tgtcands[tph].tokens[w])) == 0);
        //debug4("   word[%d] %s / %s -> %d\n",w, tgt[w].c_str(), tgtcands[tph].tokens[w]->c_str(), match);
      }
      if (match) {
        debug5("     found phrase of length %u/%u at pos %d out of %d, p=%f\n", (uint) src.size(), (uint) tgt.size(), tph, (int) tgtcands.size(), tgtcands[tph].scores[pos]);
        if (scores) {
          for (uint s=0; s<scores->size(); s++) {
            (*scores)[s]=tgtcands[tph].scores[pos+s]; // return sequence of scores
            debug2(" score[%u]: %f\n",s, (*scores)[s]);
          }
        }
        return tgtcands[tph].scores[pos];
      }
    } 
 
  } 
      
    // phrase pair wasn't found in any phrase table
    // do we have an unknown word which was copied to the target ?
  if (src.size()==1 && tgt.size()==1 && src[0]==tgt[0]) {
    debug0("     UNK: source copied to target\n");
    if (scores) {
      for (uint s=0; s<scores->size(); s++) (*scores)[s]=PROBA_COPY_UNK; // return sequence of scores
    }
    return PROBA_COPY_UNK;
  }
  
#ifdef DEBUG
  cout << "ERROR: can't find the following phrase pair in the external phrase tables: SETTING PROBA TO " << PROBA_NOT_IN_PTABLE << endl;
  for (w=0; w<src.size(); w++) cout << " " << src[w];
  cout << " |||";
  for (w=0; w<tgt.size(); w++) cout << " " << tgt[w];
  cout << " ||| " << endl;
#endif
  if (scores) {
    for (uint s=0; s<scores->size(); s++) (*scores)[s]=PROBA_NOT_IN_PTABLE; // return sequence of scores
  }
  return PROBA_NOT_IN_PTABLE;
}

/*
void PtableMosesPtree::BlockEval (Hypo &hyp, vector<string> &srcw, const int pos)
{
}
*/

void PtableMosesPtree::RescoreHyp (Hypo &hyp, vector<string> &srcw, const int pos)
{
  debug1("TGT: %s\n", hyp.trg.c_str());
  vector<string> trgw = Moses::Tokenize<std::string>(hyp.trg);

  int nws=srcw.size(), nwt=trgw.size();
  debug3("Ptable rescoring with %d source and %d target words, %d phrases\n", nws, nwt, (int) hyp.a.size());
  vector<string> srcph, trgph;  // needed to build up current phrase pair


  if (nscores>1) { 
    vector<float> res(nscores,0.0); // we request more than one score form the phrase table
    vector<float> logP(nscores,0.0); // we request more than one score form the phrase table

    for (vector<Align>::iterator al=hyp.a.begin(); al!=hyp.a.end(); al++) {
      if ((*al).se>=nws) Error("phrase table rescoring: last source word in phrase out of bounds\n");
      if ((*al).te>=nwt) Error("phrase table rescoring: last target word in phrase out of bounds\n");

      debug4("ALIGN %d-%d = %d-%d\n", (*al).sb, (*al).se, (*al).tb, (*al).te);
      srcph.clear();
      for (int w=(*al).sb; w<=(*al).se; w++) srcph.push_back(srcw[w]);
      trgph.clear();
      for (int w=(*al).tb; w<=(*al).te; w++) trgph.push_back(trgw[w]);

      GetProb(srcph,trgph,&res); // TODO: this is very inefficient, we should group together request for the same source phrase
      for (int i=0; i<nscores; i++) logP[i] += log(res[i]);
    }
    hyp.SetFeature(logP,pos);

  }
  else {
    REAL logP=0;	// we request only one score from the phrase table

    for (vector<Align>::iterator al=hyp.a.begin(); al!=hyp.a.end(); al++) {
      if ((*al).se>=nws) Error("phrase table rescoring: last source word in phrase out of bounds\n");
      if ((*al).te>=nwt) Error("phrase table rescoring: last target word in phrase out of bounds\n");

      debug4("ALIGN %d-%d = %d-%d\n", (*al).sb, (*al).se, (*al).tb, (*al).te);
      srcph.clear();
      for (int w=(*al).sb; w<=(*al).se; w++) srcph.push_back(srcw[w]);
      trgph.clear();
      for (int w=(*al).tb; w<=(*al).te; w++) trgph.push_back(trgw[w]);

      logP+=log(GetProb(srcph,trgph)); // TODO: this is very inefficient, we should group together request for the same source phrase
    }
    hyp.SetFeature(logP,pos);
  }
}
