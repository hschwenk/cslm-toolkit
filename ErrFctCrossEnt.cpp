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
#include "ErrFctCrossEnt.h"

//**********************************************************************************
// E = sum_i d_i ln o_i
REAL ErrFctCrossEnt::CalcValue(int eff_bsize) {
  REAL	*optr=output;
  REAL	*tptr=target;
  double err=0.0;

  if (eff_bsize<=0) eff_bsize=bsize;
  for (int i=0; i<eff_bsize*dim; i++) {
      err += *tptr++ * safelog(*optr++);
  }
  return (REAL) err/dim;
}

//**********************************************************************************
// 
void ErrFctCrossEnt::CalcValueBatch(int eff_bsize, REAL *res) {
  REAL	*optr=output;
  REAL	*tptr=target;

  if (eff_bsize<=0) eff_bsize=bsize;
  for (int i=0; i<eff_bsize*dim; i++) {
      *res++ = *tptr++ * safelog(*optr++);
  }
}

//**********************************************************************************
// E = sum_i d_i ln o_i

#if 0 // not used anymore
REAL ErrFctCrossEnt::CalcValueNth(int idx) {
  REAL	*optr=output + idx*dim;
  REAL	*tptr=target + idx*dim;
  double err=0.0;

  for (int i=0; i<dim; i++) {
      err += *tptr++ * safelog(*optr++);
  }
  return (REAL) err/dim;
}
#endif


//**********************************************************************************
// dE / do_i = d_i / t_i
REAL ErrFctCrossEnt::CalcGrad(int eff_bsize) {
  REAL	*optr=output;
  REAL	*tptr=target;
  REAL	*gptr=grad;
  REAL err=0.0;

  // This computes the actual gradient wrt the cross-entropy,
  // not through cross-entropy and softmax.
  // This cost should NOT be used with MachSoftmax, as MachSoftmax
  // expects the cost function to compute the gradient through the
  // softmax as well.
  if (eff_bsize<=0) eff_bsize=bsize;
  for (int i=0; i<eff_bsize*dim; i++) {
    *gptr++ = (*optr == 0) ? 0 : *tptr / *optr; // TODO
    err += *tptr++ * safelog(*optr++);
  }
  return (REAL) err/dim;
}
