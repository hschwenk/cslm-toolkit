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

#include "Lrate.h"
#include <boost/program_options/errors.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <strings.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
namespace bpo = boost::program_options;


/**
 * creates a new Lrate object corresponding to given options
 * @param sParams parameters string
 * @returns new Lrate object
 */
Lrate* Lrate::NewLrate(std::string sParams)
{
  // parameters available
  bpo::options_description od;
  od.add_options()
      ("type"  , bpo::value<std::string>()->required()   , "type of learning rate")
      ("beg"   , bpo::value<REAL>()->default_value(5E-03), "initial learning rate")
      ("mult"  , bpo::value<REAL>()->default_value(7E-08), "learning rate multiplier")
      ("min"  , bpo::value<REAL>()->default_value(1e-5)  , "learning rate minimum value")
      ("stop"  , bpo::value<REAL>()->default_value(0.0)  , "learning rate stop value")
      ("maxiter"  , bpo::value<int>()->default_value(10)  , "maximum number of iterations without improvement");

  // read parameters
  bpo::variables_map vm;
  try {
    bpo::store(
        bpo::command_line_parser(std::vector<std::string>(1, sParams)).
        extra_style_parser(Lrate::parse_params).options(od).run(), vm);
    bpo::notify(vm);
  }
  catch (bpo::error &e) {
    // error handling
    ErrorN("parsing learning rate parameters \"%s\": %s", sParams.c_str(), e.what());
    return NULL;
  }
  std::string sType = vm["type"].as<std::string>();
  REAL rBeg  = vm["beg" ].as<REAL>();
  REAL rMult = vm["mult"].as<REAL>();
  REAL rStop = vm["stop"].as<REAL>();
  REAL rMin = vm["min"].as<REAL>();
  REAL rMaxIter = vm["maxiter"].as<int>();

  // create new lrate object
  Lrate* pNewLrate = NULL;
  const char* sType_cstr = sType.c_str();
  if (strcasecmp(sType_cstr, "Decay") == 0)
    pNewLrate = new LrateExpDecay(rBeg, rMult, rStop, rMin, rMaxIter);
  else if (strcasecmp(sType_cstr, "AdaGrad") == 0)
    pNewLrate = new LrateAdaGrad(rBeg, rMult, rStop, rMin, rMaxIter);
  else if (strcasecmp(sType_cstr, "Divide") == 0)
    pNewLrate = new LrateTestAndDivide(rBeg, rMult, rStop, rMin, rMaxIter);
  else if (strcasecmp(sType_cstr, "DivideAndRecover") == 0)
    pNewLrate = new LrateDivideAndRecover(rBeg, rMult, rStop, rMin, rMaxIter);
  else
    ErrorN("parsing learning rate parameters \"%s\": unknown type '%s'", sParams.c_str(), sType.c_str());
  if (NULL == pNewLrate)
    ErrorN("parsing learning rate parameters \"%s\": can't allocate type '%s'", sParams.c_str(), sType.c_str());
  return pNewLrate;
}


/**
 * prints information about learning rate to standard output
 */
void Lrate::Info() const
{
  printf("    lower bound: %e", lrate_min);
  if (lrate_stop>0 || lrate_maxiter>0) {
     printf(", stopping");
     if (lrate_stop>0) printf(" when lrate<%e", lrate_stop);
     if (lrate_stop>0 && lrate_maxiter>0) printf(" or");
     if (lrate_maxiter>0) printf(" after %d iterations without improvement", lrate_maxiter);
  }
  printf("\n");
}


/**
 * parses parameters (type and other options)
 * @param vsTokens vector of tokens
 * @return vector of options
 * @note throws exception of class boost::program_options::error in case of error
 */
std::vector<bpo::option> Lrate::parse_params(const std::vector<std::string> &vsTokens)
{
  std::vector<bpo::option> voParsed;

  // put tokens in stream
  std::stringstream ssTokens;
  std::vector<std::string>::const_iterator iEnd = vsTokens.end();
  for (std::vector<std::string>::const_iterator iT = vsTokens.begin() ; iT != iEnd ; iT++)
    ssTokens << *iT << ' ';

  // read type (if written without parameter name)
  std::string sReadType;
  ssTokens >> sReadType;
  if (!sReadType.empty()) {
    const std::string sTypeParam("type");
    if (sTypeParam != sReadType.substr(0, sReadType.find('=')))
      voParsed.insert(voParsed.end(), bpo::option(sTypeParam, std::vector<std::string>(1, sReadType)));
    else {
      // no type without parameter name
      ssTokens.seekg(0);
      ssTokens.clear();
    }
  }

  // read other parameters
  ParseParametersLine(ssTokens, voParsed);

  // handle errors
  if (ssTokens.bad())
    throw bpo::error("internal stream error");

  return voParsed;
}


/**
 * prints information about learning rate to standard output
 */
void LrateExpDecay::Info() const
{
  printf(" - decaying learning rate: %6.2e, decay factor=%6.2e\n", lrate_beg, lrate_mult);
  Lrate::Info();
}


/**
 * updates learning rate after a forward
 * @param iNbEx number of examples seen
 */
void LrateExpDecay::UpdateLrateOnForw(ulong iNbEx)
{
  lrate = lrate_beg / (1.0 + iNbEx * lrate_mult); // quadratic decrease
  if (lrate<lrate_min) lrate=lrate_min;
}


/**
 * prints information about learning rate to standard output
 */
void LrateTestAndDivide::Info() const
{
  printf(" - learning rate: %6.2e, multiplied by %6.2e if the error increases on the development data\n", lrate, lrate_mult);
  Lrate::Info();
}


/**
 * updates learning rate after a cross-validation
 * @param rErrDev current average error
 * @param rBestErrDev best average error
 * @param sBestFile name of best machine file
 * @param pMach pointer to machine object
 * @returns true if performance is better
 */
bool LrateTestAndDivide::UpdateLrateOnDev(REAL rErrDev, REAL rBestErrDev, const char*, Mach*&)
{
  if (rErrDev < rBestErrDev) {
    lrate_iter_nogain=0;
    return true;
  }

  lrate_iter_nogain++;
  lrate *= lrate_mult;
  if (lrate<lrate_min) lrate=lrate_min;
  printf(" - multiplying learning rate by %e, new value is %e, %d iterations without improvement\n", lrate_mult, lrate, lrate_iter_nogain);
  return false;
}


/**
 * updates learning rate after a cross-validation
 * @param rErrDev current average error
 * @param rBestErrDev best average error
 * @param sBestFile name of best machine file
 * @param pMach pointer to machine object which will be reloaded if performance decrease
 * @returns true if performance is better
 */
bool LrateDivideAndRecover::UpdateLrateOnDev(REAL rErrDev, REAL rBestErrDev, const char* sBestFile, Mach*& pMach)
{
  if (LrateTestAndDivide::UpdateLrateOnDev(rErrDev, rBestErrDev, sBestFile, pMach)) {
    return true;
  }
  else {
    printf(" - reloading previous best machine ... ");
    std::ifstream ifs;
    ifs.open(sBestFile, std::ios::binary);
    if (!ifs)
      // previous best machine not available
      printf("error: %s\n", strerror(errno));
    else {
      // reload previous best machine parameters
      Mach::SetShareOffs(random()); // use a new shareOffs since we have one globale table
      Mach* pPrevMach = Mach::Read(ifs);
      Mach::SetShareOffs(0); // reset
      if (   (pMach->GetNbForw () >= pPrevMach->GetNbForw ())
          && (pMach->GetNbBackw() >= pPrevMach->GetNbBackw())
          &&  pMach->CopyParams(pPrevMach)  )
        printf("done\n");
      else
        // the machine file has been changed outside
        printf("error: the best machine file has changed\n");
      delete pPrevMach;
    }
    ifs.close();
    return false;
  }
}
