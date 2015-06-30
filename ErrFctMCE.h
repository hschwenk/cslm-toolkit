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
 * Class definition of ``mean classification error'' function (MCE)
 *   we use MSE for training, but the value of the error function is
 *   the number of wrongly classified examples
 */

#ifndef _ErrFctMCE_h
#define _ErrFctMCE_h

#include <iostream>
#include "Tools.h"
#include "ErrFct.h"


class ErrFctMCE : public ErrFct
{
public:
  ErrFctMCE(Mach &mach) : ErrFct(mach) {};
  virtual REAL CalcValue(int=0);		// Calculate value of error function = sum over all examples in minibatch
  virtual void CalcValueBatch(int, REAL*);	// Calculate value of error function, returns array for all values in mini batch
						//   (the vector must be allocated by the caller)	
  virtual REAL CalcGrad(int=0);			// calculate NEGATIF gradient of error function
};

#endif
