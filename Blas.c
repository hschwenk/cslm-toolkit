/*
 * This file is part of the continuous space language model toolkit for large
 * vocabulary speech recognition and statistical machine translation.
 *
 * Copyright 2014, Holger Schwenk, LIUM, University of Le Mans, France
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
 * $Id: Blas.c,v 1.4 2014/02/03 15:35:59 coetmeur Exp $
 *
 */

#include <math.h>

// basic implementation of fucntions on vectors
// It is more efficient to use vectorized functions, for instance those available in MKL
//

void atlas_vtanh(int *n, float *d) {int i;  for (i=0; i<*n; i++, d++) *d = tanh(*d); } 
void atlas_vlog(int *n, float *d) {int i;  for (i=0; i<*n; i++, d++) *d = log(*d); } 
void atlas_vexp(int *n, float *d) {int i;  for (i=0; i<*n; i++, d++) *d = exp(*d); } 
void atlas_vsqr(int *n, float *d) {int i;  for (i=0; i<*n; i++, d++) *d *= *d; } 

