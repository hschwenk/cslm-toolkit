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
 * Basic functions to process one hypothesis
 */


#ifndef _HYPO_H_
#define _HYPO_H_

using namespace std;

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "Tools.h"
#include "Toolsgz.h"

#define NBEST_DELIM "|||"
#define NBEST_DELIM2 " ||| "

class PtableMosesPtree;  // forward declaration for friendship in hypo

class Align {
 public:
  int sb;	// begining of source phrase
  int se;	//      end of source phrase
  int tb;	// begining of target phrase
  int te;	//      end of target phrase
 public:
  Align(int p1, int p2, int p3, int p4) : sb(p1), se(p2), tb(p3), te(p4) {};
  void Print(outputfilestream&);
};
  
class Hypo {
protected:
  int id;
  string trg;		// translation
  vector<float> f;	// feature function scores
  vector<string> trgw;	// translation segmented into words
  float	s;		// global score
  vector<Align>	a;	// alignments
  string extra;		// additonal fields at the end of the line which are preserved
  vector<REAL> p_aux;  //Aux data
  int aux_dim;
public:
  Hypo() {};
  ~Hypo() {};

  //Hypo(int p_id,string p_trg, vector<float> &p_f, float p_s) : id(p_id),trg(p_trg),f(p_f),s(p_s) { a.clear(); extra.clear(); }
  Hypo(int p_id,string p_trg, vector<float> &p_f, float p_s, vector<REAL>& paux , int auxdim =0) : id(p_id),trg(p_trg),f(p_f),s(p_s), p_aux(paux), aux_dim(auxdim){ 
	a.clear(); extra.clear();
  }

  Hypo(int p_id,string p_trg, vector<float> &p_f, float p_s, vector<Align> &p_a, vector<REAL>& paux, int auxdim =0) : id(p_id),trg(p_trg),f(p_f),s(p_s), a(p_a), p_aux(paux), aux_dim(auxdim){ 
	extra.clear();
  }
  Hypo(int p_id,string p_trg, vector<float> &p_f, float p_s, string &p_e, vector<REAL>& paux,int auxdim =0) : id(p_id),trg(p_trg),f(p_f),s(p_s), extra(p_e),  p_aux(paux), aux_dim(auxdim){ 
	a.clear();
  }

  float CalcGlobal(Weights&);
  void AddID(int o) {id+=o;};
  void Write(outputfilestream&);
  bool operator< (const Hypo&) const;
  // bool CompareLikelihoods (const Hypo&, const Hypo&) const;
  void SetFeature(float val, const int pos) {if(pos>0) f[pos-1]=val; else f.push_back(val); }
  void AddFeature(float val, const int pos) {f[pos-1] +=val;}
  void SetFeature(vector<float> &values, const int pos)
  {
    if (pos>0) { // replace existing scores (bound checks were done before)
      uint s=values.size();
      for (uint p=0; p<s; p++) f[pos-1+p]=values[p];
    }
    else { // append all the scores
      for (vector<float>::iterator i=values.begin(); i!=values.end(); i++) f.push_back(*i);
    }
  }
  void AddFeature(vector<float> &values, const int pos)
  {
    uint s=values.size();
    for (uint p=0; p<s; p++) f[pos-1+p]+=values[p];
  }
  const char *GetCstr() {return trg.c_str(); }
  vector<REAL>& GetAuxData() { return p_aux;}  
  int GetAuxDim() { return aux_dim;}
  int NbPhrases() {return a.size(); }
  friend class PtableMosesPtree;
  friend class NBest;
};

#endif
