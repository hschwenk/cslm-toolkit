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
#include <unistd.h>
#include <time.h>

#include "Tools.h"
#include "ErrFctMCE.h"

//**************************************************************************************

REAL ErrFctMCE::CalcValue(int eff_bsize) {
  REAL	*optr=output;
  REAL	*tptr=target;
  int nb_err=0;

  if (eff_bsize<=0) eff_bsize=bsize;
  for (int b=0; b<eff_bsize; b++) {
    REAL omax=optr[0], tmax=tptr[0];
    int oidx=0, tidx=0;
    for (int i=0; i<dim; i++) {
      if (*optr > omax) {omax=*optr; oidx=i;}
      if (*tptr > tmax) {tmax=*tptr; tidx=i;}
//printf("%f %f\n", *optr, *tptr);
      optr++; tptr++;
    }
    if (oidx!=tidx) nb_err++;
//printf("b=%d, oidx=%d, tidx=%d, err=%d\n", b, oidx, tidx, nb_err);
  }
  
  return (REAL) nb_err;
}

//**************************************************************************************

void ErrFctMCE::CalcValueBatch(int eff_bsize, REAL *res) {
  REAL	*optr=output;
  REAL	*tptr=target;

  if (eff_bsize<=0) eff_bsize=bsize;
  for (int b=0; b<eff_bsize; b++) {
    REAL omax=optr[0], tmax=tptr[0];
    int oidx=0, tidx=0;
    for (int i=0; i<dim; i++) {
      if (*optr > omax) {omax=*optr; oidx=i;}
      if (*tptr > tmax) {tmax=*tptr; tidx=i;}
      optr++; tptr++;
    }
    *res++ = (oidx == tidx) ? 1 : 0;
  }
}


//**************************************************************************************

#if 0 // not used any more use CalcValueBatch instead
REAL ErrFctMCE::CalcValueNth(int idx) {
  REAL	*optr=output + idx*dim;
  REAL	*tptr=target + idx*dim;

  REAL omax=optr[0], tmax=tptr[0];
  int oidx=0, tidx=0;
  for (int i=0; i<dim; i++) {
    if (*optr > omax) {omax=*optr; oidx=i;}
    if (*tptr > tmax) {tmax=*tptr; tidx=i;}
//printf("%f %f\n", *optr, *tptr);
    optr++; tptr++;
  }

  return (oidx!=tidx) ? 1.0 : 0.0;
}
#endif


//**************************************************************************************
REAL ErrFctMCE::CalcGrad(int eff_bsize) {
  REAL	*optr=output;
  REAL	*tptr=target;
  REAL	*gptr=grad;
  int nb_err=0;

  if (eff_bsize<=0) eff_bsize=bsize;

  for (int b=0; b<eff_bsize; b++) {
    REAL omax=optr[0], tmax=tptr[0];
    int oidx=0, tidx=0;
    for (int i=0; i<dim; i++) {
      if (*optr > omax) {omax=*optr; oidx=i;}
      if (*tptr > tmax) {tmax=*tptr; tidx=i;}
      *gptr++ = -(*optr++ - *tptr++);
    }
    if (oidx!=tidx) nb_err++;
  }
  return (REAL) nb_err;
}
