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
 */


#include "NBest.h"
#include "Tools.h"

#include <sstream>
#include <algorithm>

// blocks separated by '|||'
//  0:	sentence id
//  1:	hypthesis
//  2:	feature functions
//  3:	global score
//  4:	phrase alignments, e.g. 0-1=0-1 2-4=2-3 5=4

bool NBest::ParseLine(inputfilestream& inpf, inputfilestream& auxf, const int n, const bool need_alignments, const int aux_dim)
{
  static string line; // used internally to buffer an input line
  static int prev_id=-1; // used to detect a change of the n-best ID
  int new_id;
  vector<float> f;
  vector<string> blocks;
  static REAL* aux_data=NULL;
  REAL AuxValue;
  vector<REAL> aux_data_vec;
  
  if (line.empty()) {
     getline(inpf,line);
     if (inpf.eof()) return false;
     if (0 < aux_dim)
     {
        if (!auxf)  Error("Not enough auxiliary data available");
    	for (int i = 0 ; i<aux_dim ; i++)
    	{
	        auxf >> AuxValue;
                aux_data_vec.push_back(AuxValue);	
                if (auxf.eof()) return false;
    	}
    }
  }
  else {
	if (aux_data) 
	{ 
        	for (int i = 0 ; i<aux_dim ; i++)
        	{
            		aux_data_vec.push_back(aux_data[i]);
         	}
  	}
  }

  debug1("NBest::ParseLine(): %s\n", line.c_str());
    // split line into blocks
  //cerr << "PARSE line: " << line << endl;
  uint pos=0, epos;
  //while ((epos=line.find(NBEST_DELIM,pos))!=string::npos) {
  while ((epos=line.find(NBEST_DELIM,pos))<100000) {
    blocks.push_back(line.substr(pos,epos-pos));
    //cerr << " block from " << pos << " to " << epos << " : " <<  blocks.back() << endl;
    pos=epos+strlen(NBEST_DELIM);
  }
  blocks.push_back(line.substr(pos,line.size()));
  // cerr << " block: " << blocks.back() << endl;

  if (blocks.size()<4) {
    cerr << "ERROR: can't parse the following line (skipped)" << endl << line << endl;
    line.clear(); // force read of new line
    return true;
  }

  if (need_alignments && blocks.size()<5) {
    Error("alignments are needed when rescoring phrase-tables");
  }

    // parse ID
  new_id=Scan<int>(blocks[0]);
  if (prev_id>=0 && new_id!=prev_id) {
      if (!aux_data) aux_data = new REAL[aux_dim];
      int j=0;
      for (vector<REAL>::iterator x = aux_data_vec.begin(); x != aux_data_vec.end(); x++) {
         aux_data[j]= *x;
	 j++;
      }
      prev_id=new_id; return false;
  } // new nbest list has started
  prev_id=new_id;
  id=new_id;
  //cerr << "same ID " << id << endl;

  if (n>0 && nbest.size() >= (uint) n) {
    //cerr << "skipped" << endl;
    line.clear();
    return true; // skip parsing of unused hypos
  }

    // parse feature function scores
  //cerr << "PARSE features: '" << blocks[2] << "' size: " << blocks[2].size() << endl;
  pos=blocks[2].find_first_not_of(' ');
  while (pos<blocks[2].size() && (epos=blocks[2].find(" ",pos))!=string::npos) {
    string feat=blocks[2].substr(pos,epos-pos);
    //cerr << " feat: '" << feat << "', pos: " << pos << ", " << epos << endl;
    if (feat.find(":",0)!=string::npos || feat.find("=",0)!=string::npos) {
      // skip feature names (old or new Moses style)
      //cerr << "  name: " << feat << endl;
    }
    else { 
      f.push_back(Scan<float>(feat));
      //cerr << "  value: " << f.back() << endl;
    }
    pos=blocks[2].find_first_not_of(' ',epos+1);
  }
  //cerr << " FOUND " << f.size() << " features" << endl;

#ifdef BOLT_NBEST
  if (blocks.size()>4) { // copy all additional fields to the output
    string extra_info;
    for (size_t bb=4; bb<blocks.size(); bb++) {
      extra_info.append(NBEST_DELIM);
      extra_info.append(blocks[bb]);
    }
    nbest.push_back(Hypo(id, blocks[1], f, Scan<float>(blocks[3]), extra_info, aux_data_vec, aux_dim) );
  }
  else {
    nbest.push_back(Hypo(id, blocks[1], f, Scan<float>(blocks[3]), aux_data_vec, aux_dim) );
  }
#else
    // eventually parse segmentation
  if (blocks.size()>4) {
    vector<Align> a;
    pos=blocks[4].find_first_not_of(' ');

    debug1("parsing alignment in: %s\n", blocks[4].c_str());
    blocks[4].append(" "); // simplifies parsing

    //while (pos<blocks[4].size() && (epos=blocks[4].find(" ",pos))!=string::npos) // does not work !?
    while (pos<blocks[4].size() && (epos=blocks[4].find(" ",pos)) < 100000)
    {
      string align_txt=blocks[4].substr(pos,epos-pos);

      debug1(" parsing alignmnent %s:\n",align_txt.c_str());
      uint tpos=align_txt.find('=');
      if (tpos>align_txt.size()) {cerr << align_txt; Error("format error in alignment (no target phrase)"); }

      uint pos2;
      int sb,se,tb,te;
      pos2=align_txt.rfind('-',tpos);
      if (pos2>align_txt.size()) {
        debug2(" src: pos %d-%d\n",0,tpos);
        se=sb=Scan<int>(align_txt.substr(0,tpos));
      }
      else {
        debug2(" sb: pos %d-%d\n",0,pos2);
        sb=Scan<int>(align_txt.substr(0,pos2));
        pos=pos2+1; pos2=align_txt.find('=',pos);
        debug2(" se: pos %d-%d\n",pos,pos2);
        if (pos2>align_txt.size())  {cerr << align_txt; Error("format error in alignment (end of source phrase)"); }
        se=Scan<int>(align_txt.substr(pos,pos2-pos));
      }

      tpos++;
      pos2=align_txt.find('-',tpos);
      if (pos2>align_txt.size()) {
        debug1(" tgt: pos %d\n",tpos);
        te=tb=Scan<int>(align_txt.substr(tpos));
      }
      else {
        debug2(" tb: pos %d-%d\n",tpos,pos2);
        tb=Scan<int>(align_txt.substr(tpos,pos2-tpos));
        te=Scan<int>(align_txt.substr(pos2+1));
      }

      if (sb<0 || se<0 || tb<0 || te<0 || sb>se || tb>te)  {cerr << align_txt; Error("wrong numbers in alignment"); }
      debug4(" result %d-%d = %d-%d\n", sb,se,tb,te);
      a.push_back(Align(sb,se,tb,te));

      pos=blocks[4].find_first_not_of(' ',epos+1);
    }

    debug1("found %d phrases\n",(int) a.size());
    nbest.push_back(Hypo(id, blocks[1], f, Scan<float>(blocks[3]), a, aux_data_vec, aux_dim) );
  }
  else {
    nbest.push_back(Hypo(id, blocks[1], f, Scan<float>(blocks[3]), aux_data_vec, aux_dim) );
  }
#endif

  line.clear(); // force read of new line
  return true;
}


NBest::NBest(inputfilestream &inpf, inputfilestream &auxf, const int n, const bool need_alignments, const int aux_dim) 
  : max_req(262144), nreq(0), nb_diff_align(0)
{
  debug0("NBEST: constructor called\n");
  areq = new AlignReq[max_req];
  //areq.reserve(max_req);
  while (ParseLine(inpf, auxf, n, need_alignments, aux_dim));
}


NBest::~NBest()
{
  debug0("NBEST: destructor called\n");
  nbest.clear();
  srcw.clear();
  if (areq) delete [] areq;
  //areq.clear();
}

void NBest::Write(outputfilestream &outf, int n)
{
  if (n<1 || (uint) n>nbest.size()) n=nbest.size();
  for (int i=0; i<n; i++) nbest[i].Write(outf);
}


void NBest::CalcGlobal(Weights &w)
{
  //cerr << "NBEST: calc global of size " << nbest.size() << endl;
  for (vector<Hypo>::iterator i = nbest.begin(); i != nbest.end(); i++) {
    (*i).CalcGlobal(w);
  }
}


void NBest::Sort() {
  sort(nbest.begin(),nbest.end());
}


void NBest::AddID(const int o)
{
  for (vector<Hypo>::iterator i = nbest.begin(); i != nbest.end(); i++) {
    (*i).AddID(o);
  }
}

void NBest::RescoreLM(NbestLM &lm, const int lm_pos)
{
  for (vector<Hypo>::iterator i = nbest.begin(); i != nbest.end(); i++) {
    lm.RescoreHyp(*i,lm_pos);
  }
  lm.FinishPending();
}

#undef OLD
#ifdef OLD
void NBest::RescorePtable(PtableMosesPtree &pt, ifstream &srcf, const int tm_pos)
{
    // get a source line and segment into words
  string src;
  getline(srcf,src);
  if (srcf.eof())
    ErrorN("EOF in source text for n-best hypothesis id=%d", id);

  srcw.clear();
  srcw = Moses::Tokenize<std::string>(src);

  for (vector<Hypo>::iterator i = nbest.begin(); i != nbest.end(); i++) {
     pt.RescoreHyp(*i,srcw,tm_pos);
  }
}
#else

void NBest::RescorePtable(PtableMosesPtree &pt, ifstream &srcf, const int tm_pos)
{
    // get a source line and segment into words
  string src;
  getline(srcf,src);
  if (srcf.eof())
    ErrorN("EOF in source text for n-best hypothesis id=%d", id);

  srcw.clear();
  srcw = Moses::Tokenize<std::string>(src);

  int nscores = pt.GetNscores();
  debug2("NBest::RescorePtable(): %d scores at position %d\n", nscores, tm_pos);
  debug2("SRC with %d words: %s\n", (int) srcw.size(),  src.c_str());

  vector<float> null_scores(nscores, 0.0);

  for (vector<Hypo>::iterator hi=nbest.begin(); hi!= nbest.end(); hi++) {
      // reset the features that will be modified in BlockFinish()
      // we already append them here if requested
    if (nscores>1) (*hi).SetFeature(null_scores, tm_pos);
              else (*hi).SetFeature(0.0, tm_pos);
    
    hi->trgw = Moses::Tokenize<std::string>(hi->trg);
    for (vector<Align>::iterator ali=(*hi).a.begin(); ali!=(*hi).a.end(); ali++) {
      areq[nreq].sb = (*ali).sb;
      areq[nreq].se = (*ali).se;
      for (int w=(*ali).tb; w<=(*ali).te; w++) areq[nreq].tgph.push_back((*hi).trgw[w]);
      areq[nreq].hyp=&(*hi);
      if (++nreq >= max_req) BlockFinish(pt,tm_pos);
    }
  }
  BlockFinish(pt,tm_pos);
}
#endif

void NBest::RescorePtableInv(PtableMosesPtree &pt, ifstream &srcf, const int tm_pos)
{
  Error("NBest::RescorePtableInv");
    // get a source line and segment into words
  string src;
  getline(srcf,src);
  if (srcf.eof())
    ErrorN("EOF in source text for n-best hypothesis id=%d", id);

  srcw.clear();
  srcw = Moses::Tokenize<std::string>(src);

  int nscores = pt.GetNscores();
  debug2("NBest::RescorePtable(): %d scores at position %d\n", nscores, tm_pos);
  debug2("SRC with %d words: %s\n", (int) srcw.size(),  src.c_str());

  vector<float> null_scores(nscores, 0.0);

  for (vector<Hypo>::iterator hi=nbest.begin(); hi!= nbest.end(); hi++) {
      // reset the features that will be modified in BlockFinish()
      // we already append them here if requested
    if (nscores>1) (*hi).SetFeature(null_scores, tm_pos);
              else (*hi).SetFeature(0.0, tm_pos);
    
    hi->trgw = Moses::Tokenize<std::string>(hi->trg);
    for (vector<Align>::iterator ali=(*hi).a.begin(); ali!=(*hi).a.end(); ali++) {
      areq[nreq].sb = (*ali).sb;
      areq[nreq].se = (*ali).se;
      for (int w=(*ali).tb; w<=(*ali).te; w++) areq[nreq].tgph.push_back((*hi).trgw[w]);
      areq[nreq].hyp=&(*hi);
      if (++nreq >= max_req) BlockFinish(pt,tm_pos);
    }
  }
  BlockFinish(pt,tm_pos);
}

  // compare source and target phrases
int AlignReqComp(const void *v1, const void *v2)
{
  AlignReq* a1=(AlignReq*) v1, *a2=(AlignReq*) v2;

  if (a1->sb < a2->sb) return -1;
  if (a1->sb > a2->sb) return  1;
  if (a1->se < a2->se) return -1;
  if (a1->se > a2->se) return  1;
  if (a1->tgph.size() < a2->tgph.size()) return -1;
  if (a1->tgph.size() > a2->tgph.size()) return  1;
  for (int w=0; w<(int)a1->tgph.size(); w++) {
    if (a1->tgph[w] < a2->tgph[w]) return -1;
    if (a1->tgph[w] > a2->tgph[w]) return  1;
  }

  return 0; // both are equal
}

  // compare source phrases only
int AlignReqCompSrc(const void *v1, const void *v2)
{
  AlignReq* a1=(AlignReq*) v1, *a2=(AlignReq*) v2;

  if (a1->sb < a2->sb) return -1;
  if (a1->sb > a2->sb) return  1;
  if (a1->se < a2->se) return -1;
  if (a1->se > a2->se) return  1;

  return 0; // both are equal
}
  

float NBest::GetAlignProb(PtableMosesPtree &pt, AlignReq &aq, const int tm_pos, vector<float> *logP_v) // TODO: param tm_pos is unused
{
  debug1("TGT: %s\n", aq.hyp->trg.c_str());
  debug4("ALIGN %d-%d = %s-%s\n", aq.sb, aq.se, aq.tgph[0].c_str(), aq.tgph.back().c_str());

  if (aq.se >= (int) srcw.size()) Error("phrase table rescoring: last source word in phrase is out of bounds\n");

    // build up current source phrase pair, TODO: switch to reference ?
  vector<string> srcph;
  for (int w=aq.sb; w<=aq.se; w++) srcph.push_back(srcw[w]);

  //printf("get Prob for %s..%s || %s..%s  -> %f\n",srcw[0].c_str(),srcw.back().c_str(),trgw[0].c_str(),trgw.back().c_str,pt.GetProb(srcph,trgph));
  //printf("ALIGN %d-%d = %s-%s -> P=%f\n",aq.sb,aq.se,aq.tb,aq.te,pt.GetProb(srcph,trgph));
  if (logP_v) {
    pt.GetProb(srcph,aq.tgph,logP_v);
    for (vector<float>::iterator fi=logP_v->begin(); fi!=logP_v->end(); fi++) *fi = log(*fi);
    return (*logP_v)[0];
  }
  else {
    return log(pt.GetProb(srcph,aq.tgph));
  }
}

void NBest::BlockFinish(PtableMosesPtree &pt, int tm_pos)
{
  debug2("BlockFinish(): processing %d delayed requests, source: %d words\n", nreq, (int)srcw.size());

  if (nreq==0) return;

  qsort(areq, nreq, sizeof(AlignReq), AlignReqComp);

  int nscores = pt.GetNscores();
  int cnt=1;

  if (tm_pos==0) tm_pos=areq[0].hyp->f.size()-nscores+1; // correct position in append mode
  debug2("cumulating %d scores starting at position %d\n", nscores, tm_pos);

    // request phrase probas for the first alignment
  if (nscores>1) {
    vector<float> logP_scores(nscores, 0.0);
    debug4("request align 0: %d-%d %s-%s (several scores)\n",areq[0].sb,areq[0].se,areq[0].tgph[0].c_str(),areq[0].tgph.back().c_str());
    GetAlignProb(pt,areq[0],tm_pos, &logP_scores);
    areq[0].hyp->AddFeature(logP_scores,tm_pos);

    for (int n=1; n<nreq; n++) {
      if (AlignReqComp(areq+n-1, areq+n) != 0) {
          // new alignment pair -> calculate new logP
        debug5("request align %d: %d-%d %s-%s\n", cnt,areq[n].sb,areq[n].se,areq[n].tgph[0].c_str(),areq[n].tgph.back().c_str());
        GetAlignProb(pt,areq[n],tm_pos, &logP_scores);
        cnt++;
      }
      //printf("add %f to hyp %s\n",logP,areq[n].hyp->trg.c_str());
      areq[n].hyp->AddFeature(logP_scores,tm_pos);	// cumulate
    }
  }
  else {
    debug4("request align 0: %d-%d %s-%s\n",areq[0].sb,areq[0].se,areq[0].tgph[0].c_str(),areq[0].tgph.back().c_str());
    float logP = GetAlignProb(pt,areq[0],tm_pos);
    areq[0].hyp->AddFeature(logP,tm_pos);

    for (int n=1; n<nreq; n++) {
      if (AlignReqComp(areq+n-1, areq+n) != 0) {
          // new alignment pair -> calculate new logP
        debug5("request align %d: %d-%d %s-%s\n", cnt,areq[n].sb,areq[n].se,areq[n].tgph[0].c_str(),areq[n].tgph.back().c_str());
        logP = GetAlignProb(pt,areq[n],tm_pos);
        cnt++;
      }
      //printf("add %f to hyp %s\n",logP,areq[n].hyp->trg.c_str());
      areq[n].hyp->AddFeature(logP,tm_pos);	// cumulate
    }
  }

  debug1(" %d different alignments\n", cnt);
  nb_diff_align += cnt;
}

int NBest::NbPhrases()
{
  int cnt=0;
  for (vector<Hypo>::iterator i = nbest.begin(); i != nbest.end(); i++) {
    cnt += (*i).NbPhrases();
  }

  return cnt;
}

//**********************************************************
//
// caching algorithm for TM rescoring with CSTM
//
//**********************************************************


// this is identical to Moses ptable rescoring, we just call a different BlockFinish
void NBest::RescorePtable(NbestCSTM &cstm, ifstream &srcf, const int tm_pos)
{
    // get a source line and segment into words
  string src;
  getline(srcf,src);
  if (srcf.eof())
    ErrorN("EOF in source text for n-best hypothesis id=%d", id);

  srcw.clear();
  srcw = Moses::Tokenize<std::string>(src);

  debug1("NBest::RescorePtable(): CSTM score at position %d\n", tm_pos);
  debug2("SRC with %d words: %s\n", (int) srcw.size(),  src.c_str());

  for (vector<Hypo>::iterator hi=nbest.begin(); hi!= nbest.end(); hi++) {
      // reset the feature that will be modified in BlockFinish()
      // we already append it here if requested
    (*hi).SetFeature(0.0, tm_pos);
    
    hi->trgw = Moses::Tokenize<std::string>(hi->trg);
    int nw=(int) hi->trgw.size();
    debug2("CSTM token target: %s  %d words\n", hi->trg.c_str(), nw);
    for (vector<Align>::iterator ali=(*hi).a.begin(); ali!=(*hi).a.end(); ali++) {
      areq[nreq].sb = (*ali).sb;
      areq[nreq].se = (*ali).se;
      debug5("CSTM process areq %d, src: %d-%d, tgt: %d-%d\n",nreq,(*ali).sb,(*ali).se,(*ali).tb,(*ali).te);
      if ((*ali).tb<0 || (*ali).tb>=nw || ((*ali).te<0 || (*ali).te>=nw)) {
        fprintf(stderr,"skipping line with targets out of bound in alignment %d-%d=%d-%d\n",(*ali).sb,(*ali).se,(*ali).tb,(*ali).te);
        continue;
      }
      for (int w=(*ali).tb; w<=(*ali).te; w++) areq[nreq].tgph.push_back((*hi).trgw[w]);
      cstm.LookupTarget(areq[nreq].tgph, areq[nreq].tgwid); // TODO: this is inefficient, the same target will appear many times
      areq[nreq].hyp=&(*hi);
      if (++nreq >= max_req) BlockFinish(cstm,tm_pos);
    }
  }
  BlockFinish(cstm,tm_pos);
}

// this is identical to Moses ptable rescoring, we just call a different BlockFinish
void NBest::RescorePtableInv(NbestCSTM &cstm, ifstream &srcf, const int tm_pos)
{
  Error("NBest::RescorePtableInv()");
}

void NBest::BlockFinish(NbestCSTM &cstm, int tm_pos)
{
  debug2("BlockFinish(): processing %d delayed requests, source: %d words\n", nreq, (int)srcw.size());

  if (nreq==0) return;
  int bsize=cstm.mach->GetBsize();

  qsort(areq, nreq, sizeof(AlignReq), AlignReqComp);

  if (tm_pos==0) tm_pos=areq[0].hyp->f.size(); // correct position in append mode
  debug1("cumulating 1 score starting at position %d\n", tm_pos);

  vector<string> srcph;				// one source phrase
  vector< vector<string> > src_phrases;		// all possible source phrase in this block, size
  
    // process first phrase pair
  areq[0].bs=0;
  cstm.AddToInput(0,srcw,areq[0].sb,areq[0].se);
  srcph.clear();
  for (int w=areq[0].sb; w<=areq[0].se; w++) srcph.push_back(srcw[w]);
  src_phrases.push_back(srcph);

  int cnt=1;

  int req_beg=0;	// start of current CSLM block in large request array
  int bs=0;             // current block index in forward bunch

  for (int n=1; n<nreq; n++) {
    if (AlignReqCompSrc(areq+n-1, areq+n) != 0) { // new source phrase 
        // first process bunch if full
      bs++;
      debug1("   %d new context\n", bs);
      if (bs >= bsize) {
        cstm.trainer->ForwAndCollect(src_phrases,areq,req_beg,n-1,bs,tm_pos);
        bs=0; req_beg=n;
      }
          // add new source phrase to bunch for forward pass
          // REMARK: this is not perfect since some of the examples may be out of slist and we actually wouldn't
          //         need a forward pass for them. However, all request of an n-best block must be performed before
          //         we go to the next n-best block, In practice there are often less than 128 difference source phrases.
          //         Therefore, we only do one forward pass anyway
      areq[n].bs=bs;
      cstm.AddToInput(bs,srcw,areq[n].sb,areq[n].se);
      srcph.clear();
      for (int w=areq[n].sb; w<=areq[n].se; w++) srcph.push_back(srcw[w]);
      src_phrases.push_back(srcph);
      cnt++;
    }
    else
      areq[n].bs=bs;
  }
  cstm.trainer->ForwAndCollect(src_phrases,areq,req_beg,nreq-1,bs+1,tm_pos);
  // FreeReq(); TODO

  printf(" %d different source phrases\n", cnt);
  nb_diff_align += cnt;
}

