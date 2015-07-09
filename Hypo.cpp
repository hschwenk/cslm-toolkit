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


#include "Hypo.h"
#include "Tools.h"

#include <iostream>

void Align::Print(outputfilestream &outf)
{
  if (sb==se) outf << sb;
         else outf << sb << "-" << se;
  outf << "=";
  if (tb==te) outf << tb;
         else outf << tb << "-" << te;
}

void Hypo::Write(outputfilestream &outf)
{
  outf << id << NBEST_DELIM2 << trg << NBEST_DELIM2;
  for (vector<float>::iterator i = f.begin(); i != f.end(); i++)
    outf << (*i) << " ";
  outf << NBEST_DELIM << " " << s;

  if (a.size()>0) {
    outf << " " << NBEST_DELIM;
    for (vector<Align>::iterator i = a.begin(); i != a.end(); i++) {
      outf << " "; (*i).Print(outf);
    }
  }

#ifdef BOLT_NBEST
    outf << " " << extra;
#endif
  
  outf << endl;
}

float Hypo::CalcGlobal(Weights &w)
{
  debug0("HYP: calc global\n");

  uint sz=w.val.size();
  if (sz<f.size()) {
    cerr << " - NOTE: padding weight vector with " << f.size()-sz << " zeros" << endl;
    w.val.resize(f.size());
    for (uint i=sz; i<w.val.size(); i++) w.val[i]=0;
  }

  s=0;
  debug0(" scores:");
  for (uint i=0; i<f.size(); i++) {
    debug2(" %f x %e", w.val[i], f[i]);
    s+=w.val[i]*f[i];
  }
  debug1(" -> global score %e\n", s);

  return s;
}

// this is actually a "greater than" since we want to sort in descending order
bool Hypo::operator< (const Hypo &h2) const {
  return (this->s > h2.s);
}

