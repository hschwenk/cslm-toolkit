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

#ifndef _Lrate_h_
#define _Lrate_h_

#include "Mach.h"
#include "Tools.h"
#include <boost/program_options/option.hpp>
#include <sys/types.h>
#include <string>
#include <vector>


/**
 * abstract base class to compute learning rates
 */
class Lrate
{
public:

  /**
   * type of Lrate
   */
  enum LRateType {
    LR_Type_Base = 0,
    LR_Type_ExpDecay,
    LR_Type_AdaGrad,
    LR_Type_TestAndDivide,
    LR_Type_DivideAndRecover
  };

  /**
   * creates a new Lrate object corresponding to given options
   * @param sParams parameters string
   * @returns new Lrate object
   */
  static Lrate* NewLrate(std::string sParams);

  /**
   * destroys learning rate object
   */
  virtual ~Lrate() {}

  /**
   * gets Lrate type
   */
  virtual inline Lrate::LRateType GetType() const { return Lrate::LR_Type_Base; }

  /**
   * gets current learning rate value
   */
  inline REAL GetLrate() const { return lrate; }

  /**
   * checks if learning rate stop value is reached
   * @return true if current value is less than stop value
   */
  inline bool StopReached() const
  {
     if (lrate <= lrate_stop) {
       printf(" - minimal allowed learning rate reached\n");
       return true;
     }
     if (lrate_iter_nogain>=lrate_maxiter) {
       printf(" - no improvements after %d iterations\n", lrate_maxiter);
       return true;
     }
     return false;
  }

  /**
   * prints information about learning rate to standard output
   */
  virtual void Info() const;

  /**
   * updates learning rate after a forward
   * @param iNbEx number of examples seen
   */
  virtual inline void UpdateLrateOnForw(ulong iNbEx) {}

  /**
   * updates learning rate after a backward
   */
  virtual inline void UpdateLrateOnBackw() {}

  /**
   * updates learning rate after a cross-validation
   * @param rErrDev current average error
   * @param rBestErrDev best average error
   * @param sBestFile name of best machine file
   * @param pMach pointer to machine object which could be reloaded
   * @returns true if performance is better
   */
  virtual inline bool UpdateLrateOnDev(REAL rErrDev, REAL rBestErrDev, const char* sBestFile, Mach*& pMach)
  {
    if (rErrDev < rBestErrDev) lrate_iter_nogain=0;
                          else lrate_iter_nogain++;
    return (rErrDev < rBestErrDev);
  }


protected:

  REAL lrate;      ///< current value
  REAL lrate_beg;  ///< value at beginning
  REAL lrate_mult; ///< multiplier
  REAL lrate_stop; ///< stop value
  REAL lrate_min; ///< minimal value (lower bound)
  int lrate_maxiter; ///< maximum number of iterations without improvements
  int lrate_iter_nogain; ///< counts the number of iterations without improvements

  /**
   * creates new learning rate object
   * @param rLrateBeg learning rate value at beginning
   * @param rLrateMult learning rate multiplier
   * @param rLrateStop learning stop value
   * @param rLrateMin learning rate minimum value
   * @param rLrateMaxIter maximum number of iterations without improvement
   */
  Lrate(REAL rLrateBeg = 0.01, REAL rLrateMult = 0, REAL rLrateStop = 0, REAL rLrateMin = 1e-5, int rLrateMaxIter = 10) :
    lrate(rLrateBeg), lrate_beg(rLrateBeg), lrate_mult(rLrateMult), lrate_stop(rLrateStop), lrate_min(rLrateMin), lrate_maxiter(rLrateMaxIter), lrate_iter_nogain(0) {}


private:

  /**
   * parses parameters (type and other options)
   * @param vsTokens vector of tokens
   * @return vector of options
   * @note throws exception of class boost::program_options::error in case of error
   */
  static std::vector<boost::program_options::option> parse_params(const std::vector<std::string> &vsTokens);

};


/**
 * learning rate with exponential decay
 */
class LrateExpDecay : public Lrate
{
public:

  /**
   * creates new learning rate object
   * @param rLrateBeg learning rate value at beginning
   * @param rLrateMult learning rate multiplier
   * @param rLrateStop learning stop value
   * @param rLrateMin learning rate minimum value
   * @param rLrateMaxIter maximum number of iterations without improvement
   */
  LrateExpDecay(REAL rLrateBeg = 0.01, REAL rLrateMult = 0, REAL rLrateStop = 0, REAL rLrateMin = 1e-5, int rLrateMaxIter = 10) :
    Lrate(rLrateBeg, rLrateMult, rLrateStop, rLrateMin, rLrateMaxIter) {}

  /**
   * destroys learning rate object
   */
  virtual ~LrateExpDecay() {}

  /**
   * gets Lrate type
   */
  virtual inline Lrate::LRateType GetType() const { return Lrate::LR_Type_ExpDecay; }

  /**
   * prints information about learning rate to standard output
   */
  virtual void Info() const;

  /**
   * updates learning rate after a forward
   * @param iNbEx number of examples seen
   */
  virtual void UpdateLrateOnForw(ulong iNbEx);

};


/**
 * learning rate modified during backward
 */
class LrateAdaGrad : public Lrate
{
public:

  /**
   * creates new learning rate object
   * @param rLrateBeg learning rate value at beginning
   * @param rLrateMult learning rate multiplier
   * @param rLrateStop learning stop value
   * @param rLrateMin learning rate minimum value
   * @param rLrateMaxIter maximum number of iterations without improvement
   */
  LrateAdaGrad(REAL rLrateBeg = 0.01, REAL rLrateMult = 0, REAL rLrateStop = 0, REAL rLrateMin = 1e-5, int rLrateMaxIter = 10) :
    Lrate(rLrateBeg, rLrateMult, rLrateStop, rLrateMin, rLrateMaxIter) {}

  /**
   * destroys learning rate object
   */
  virtual ~LrateAdaGrad() {}

  /**
   * gets Lrate type
   */
  virtual inline Lrate::LRateType GetType() const { return Lrate::LR_Type_AdaGrad; }

  /**
   * updates learning rate after a backward
   */
  virtual inline void UpdateLrateOnBackw() { Lrate::UpdateLrateOnBackw(); }

};


/**
 * learning rate modified in function of the performance on the development data
 */
class LrateTestAndDivide : public Lrate
{
public:

  /**
   * creates new learning rate object
   * @param rLrateBeg learning rate value at beginning
   * @param rLrateMult learning rate multiplier
   * @param rLrateStop learning stop value
   * @param rLrateMin learning rate minimum value
   * @param rLrateMaxIter maximum number of iterations without improvement
   */
  LrateTestAndDivide(REAL rLrateBeg = 0.01, REAL rLrateMult = 0, REAL rLrateStop = 0, REAL rLrateMin = 1e-5, int rLrateMaxIter = 10) :
    Lrate(rLrateBeg, rLrateMult, rLrateStop, rLrateMin, rLrateMaxIter) {}

  /**
   * destroys learning rate object
   */
  virtual ~LrateTestAndDivide() {}

  /**
   * gets Lrate type
   */
  virtual inline Lrate::LRateType GetType() const { return Lrate::LR_Type_TestAndDivide; }

  /**
   * prints information about learning rate to standard output
   */
  virtual inline void Info() const;

  /**
   * updates learning rate after a cross-validation
   * @param rErrDev current average error
   * @param rBestErrDev best average error
   * @param sBestFile name of best machine file
   * @param pMach pointer to machine object
   * @returns true if performance is better
   */
  virtual bool UpdateLrateOnDev(REAL rErrDev, REAL rBestErrDev, const char* sBestFile, Mach*& pMach);

};


/**
 * learning rate modified in function of the performance on the development data
 * @note previous best machine is reloaded if performance decrease
 */
class LrateDivideAndRecover : public LrateTestAndDivide
{
public:

  /**
   * creates new learning rate object
   * @param rLrateBeg learning rate value at beginning
   * @param rLrateMult learning rate multiplier
   * @param rLrateStop minimum value
   * @param rLrateMin learning rate minimum value
   * @param rLrateMaxIter maximum number of iterations without improvement
   */
  LrateDivideAndRecover(REAL rLrateBeg = 0.01, REAL rLrateMult = 0, REAL rLrateStop = 0, REAL rLrateMin = 1e-5, int rLrateMaxIter = 10) :
    LrateTestAndDivide(rLrateBeg, rLrateMult, rLrateStop, rLrateMin, rLrateMaxIter) {}

  /**
   * destroys learning rate object
   */
  virtual ~LrateDivideAndRecover() {}

  /**
   * gets Lrate type
   */
  virtual inline Lrate::LRateType GetType() const { return Lrate::LR_Type_DivideAndRecover; }

  /**
   * updates learning rate after a cross-validation
   * @param rErrDev current average error
   * @param rBestErrDev best average error
   * @param sBestFile name of best machine file
   * @param pMach pointer to machine object which will be reloaded if performance decrease
   * @returns true if performance is better
   */
  virtual bool UpdateLrateOnDev(REAL rErrDev, REAL rBestErrDev, const char* sBestFile, Mach*& pMach);

};


#endif // _Lrate_h_
