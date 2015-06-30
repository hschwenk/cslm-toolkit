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

#include <boost/program_options/errors.hpp>
#include <boost/program_options/parsers.hpp>
#include <cstring>
#include <strings.h>
#include "MachAvr.h"
#include "MachConfig.h"
#include "MachJoin.h"
#include "MachLinRectif.h"
//#include "MachMax.h"	// experimental
#include "MachPar.h"
#include "MachSeq.h"
#include "MachSig.h"
#include "MachSoftmax.h"
#include "MachSoftmaxStable.h"
#include "MachSoftmaxClass.h"
#include "MachSplit.h"
#include "MachSplit1.h"
//#include "MachStab.h"
//#include "MachStacked.h"
//#include "MachTabSh.h"
#include "MachTanh.h"
#include "Tools.h"
#include "MachCopy.h"

namespace bpo = boost::program_options;

/**
 * creates a machine configuration reader
 * @param bNeedConfFile true if configuration file is required on command line, false otherwise
 * @param rInitBias general value for random initialization of the bias (0.1 by default)
 */
MachConfig::MachConfig (bool bNeedConfFile, REAL rInitBias) :
      bSelectedOptions(false),
      bHelpRequest(false),
      bNeedConfFile(bNeedConfFile),
      bReadMachOnly(false),
      iRepeat(1),
      rInitBias(rInitBias),
      eErrorCode(MachConfig::NoError),
      odCommandLine("Command line options"),
      odSelectedConfig("Configuration options")
{
  /* set general options (in command line and configuration file) */

  // general options in command line only
  this->odCommandLine.add_options()
          ("help"                 , "produce help message")
          ("config-file,c"        , bpo::value< std::vector<std::string> >(), "configuration file (can be set without option name)")
          ;
  this->podCommandLine.add("config-file", -1); // command line may contain configuration file name without option name

  // general options in configuration file and selectable for command line
  this->odGeneralConfig.add_options()
          ("mach,m"               , opt_sem<std::string>::new_sem(), "file name of the machine")
          ("src-word-list,s"      , opt_sem<std::string>::new_sem(), "word list of the source vocabulary")
          ("tgt-word-list,w"      , opt_sem<std::string>::new_sem(), "word list of the vocabulary and counts (used to select the most frequent words)")
          ("word-list,w"          , opt_sem<std::string>::new_sem(), "word list of the vocabulary and counts (used to select the most frequent words)")
          ("input-file,i"         , opt_sem<std::string>::new_sem(), "file name of the input n-best list")
          ("aux-file,a"           , opt_sem<std::string>::new_sem(), "file name of the auxiliary data")
          ("output-file,o"        , opt_sem<std::string>::new_sem(), "file name of the output n-best list")
          ("source-file,S"        , opt_sem<std::string>::new_sem(), "file name of the file with source sentences (needed for TM rescoring)")
          ("phrase-table"         , opt_sem<std::string>::new_sem(), "rescore with a Moses on-disk phrase table")
          ("phrase-table2"        , opt_sem<std::string>::new_sem(), "use a secondary Moses phrase table")
          ("test-data,t"          , opt_sem<std::string>::new_sem(), "test data")
          ("train-data,t"         , opt_sem<std::string>::new_sem(), "training data")
          ("dev-data,d"           , opt_sem<std::string>::new_sem(), "development data (optional)")
          ("lm,l"                 , opt_sem<std::string>::new_sem(), "file name of the machine (only necessary when using short lists)")
          ("output-probas"        , opt_sem<std::string>::new_sem(), "write sequence of log-probas to file (optional)")
          ("cslm,c"               , opt_sem<std::string>::new_sem(), "rescore with a CSLM")
          ("vocab,v"              , opt_sem<std::string>::new_sem(), "word-list to be used with the CSLM")
          ("cstm,C"               , opt_sem<std::string>::new_sem(), "rescore with a CSTM")
          ("vocab-source,b"       , opt_sem<std::string>::new_sem(), "source word-list to be used with the CSTM")
          ("vocab-target,B"       , opt_sem<std::string>::new_sem(), "target word-list to be used with the CSTM")
          ("weights,w"            , opt_sem<std::string>::new_sem(), "coefficients of the feature functions")
          ("tm-scores,N"          , opt_sem<std::string>::new_sem()->default_value("4:0"), "specification of the TM scores to be used (default first 4)")
          ("MachSeed,Mseed"       , opt_sem<int> ::new_sem()->default_value(0),"Machine seed for random weights init (default: do not set the seed)")
	  ("lrate,L"              , opt_sem<std::string>::new_sem()->default_value("Decay beg=5e-3 mult=7e-8 stop=0"), "learning rate applied: type (Decay AdaGrad Divide DivideAndRecover), initial value, multiplier and learning stop value")
          ("inn,I"                , opt_sem<int> ::new_sem(                     )->default_value(    0  ), "number of hypothesis to read per n-best (default all)")
          ("outn,O"               , opt_sem<int> ::new_sem(                     )->default_value(    0  ), "number of hypothesis to write per n-best (default all)")
          ("offs,a"               , opt_sem<int> ::new_sem(                     )->default_value(    0  ), "add offset to n-best ID (useful for separately generated n-bests)")
          ("aux-dim,n"            , opt_sem<int> ::new_sem(                     )->default_value(    0  ), "dimension of auxiliary data")
          ("num-scores,n"         , opt_sem<int> ::new_sem(                     )->default_value(    5  ), "number of scores in phrase table")
          ("ctxt-in,c"            , opt_sem<int> ::new_sem(                     )->default_value(    7  ), "input context size")
          ("ctxt-out,C"           , opt_sem<int> ::new_sem(                     )->default_value(    7  ), "output context size")
          ("curr-iter,C"          , opt_sem<int> ::new_sem(                     )->default_value(    0  ), "current iteration when continuing training of a neural network")
          ("last-iter,I"          , opt_sem<int> ::new_sem(                     )->default_value(   10  ), "last iteration of neural network")
          ("order"                , opt_sem<int> ::new_sem(                     )->default_value(    4  ), "order of the LM to apply on the test data (must match CSLM, but not necessarily back-off LM)")
          ("mode,M"               , opt_sem<int> ::new_sem(                     )->default_value(    3  ), "mode of the data (1=IGN_BOS 2=IGN_UNK 4=IGN_UNK_ALL, 8=IGN_EOS)")
          ("lm-pos,p"             , opt_sem<int> ::new_sem(                     )->default_value(    0  ), "position of LM score (1..n, 0 means to append it)")
          ("tm-pos,P"             , opt_sem<int> ::new_sem(                     )->default_value(    0  ), "position of the TM scores, up to 4 values")
          ("target-pos,T"         , opt_sem<int> ::new_sem(                     )->default_value(   -1  ), "position of the predicted word in the n-gram, default: last one")
          ("buf-size,b"           , opt_sem<int> ::new_sem(                     )->default_value(16384  ), "buffer size")
          ("block-size,B"         , opt_sem<int> ::new_sem(&this->iBlockSize    )->default_value(  128  ), "block size for faster training")
          ("drop-out,O"           , opt_sem<REAL>::new_sem(&this->rPercDropOut  )->default_value(    0.0), "percentage of neurons to be used for drop-out [0-1] (set by default to 0 to turn it off)")
          ("random-init-project,r", opt_sem<REAL>::new_sem(&this->rInitProjLayer)->default_value(    0.1), "value for random initialization of the projection layer")
          ("random-init-weights,R", opt_sem<REAL>::new_sem(&this->rInitWeights  )->default_value(    0.1), "value for random initialization of the weights")
          ("clip-weights,w"       , opt_sem<REAL>::new_sem(&this->rClipWeights  )->default_value(    0  ), "value for clipping weights (no clipping by default)")
          ("clip-gradients-weights,g",opt_sem<REAL>::new_sem(&this->rClipGradWeights)->default_value(0  ), "value for clipping gradients on weights (no clipping by default)")
          ("clip-gradients-bias,G", opt_sem<REAL>::new_sem(&this->rClipGradBias )->default_value(    0  ), "value for clipping gradients on biases (no clipping by default)")
          ("weight-decay,W"       , opt_sem<REAL>::new_sem(                     )->default_value(  3E-05), "coefficient of weight decay")
          ("backward-tm,V"        , opt_sem<bool>::new_sem()->zero_tokens(), "use an inverse back-ward translation model")
          ("renormal,R"           , opt_sem<bool>::new_sem()->zero_tokens(), "renormalize all probabilities, slow for large short-lists")
          ("recalc,r"             , opt_sem<bool>::new_sem()->zero_tokens(), "recalculate global scores")
          ("sort,s"               , opt_sem<bool>::new_sem()->zero_tokens(), "sort n-best list according to the global scores")
          ("lexical,h"            , opt_sem<bool>::new_sem()->zero_tokens(), "report number of lexically different hypothesis")
          ("server,X"             , opt_sem<bool>::new_sem()->zero_tokens(), "run in server mode listening to a named pipe to get weights for new solution extraction")
          ("unstable-sort,U"      , opt_sem<bool>::new_sem()->zero_tokens(), "use unstable sort (compatility mode with older version of the CSLM toolkit)")
          ("use-word-class,u"     , opt_sem<bool>::new_sem()->zero_tokens(), "use word class to structure the output layer")
          ("dump-activities,A"    , opt_sem<std::vector<std::string> >::new_sem(), "specify layer and filename to dump the activity for each n-gram (eg \"3:layer3.txt\")")
#ifdef BLAS_CUDA
          ("cuda-device,D"        , opt_sem<std::vector<std::string> >::new_sem(), "select CUDA device (eg \"0:2\" for devices 0 and 2)")
          ("cuda-dev-num,N"       , opt_sem<int>::new_sem()->default_value(1),  "number of CUDA devices to be used")
#endif
          ;


  /* set machine names */

  // machine names are defined in configuration file options to be recognized as valid options
  this->odMachineTypes.add_options()
          ("machine.Mach"         , bpo::value<std::vector<std::string> >())
          ("machine.Tab"          , bpo::value<std::vector<std::string> >())
          ("machine.Linear"       , bpo::value<std::vector<std::string> >())
          ("machine.LinRectif"    , bpo::value<std::vector<std::string> >())
          ("machine.Sig"          , bpo::value<std::vector<std::string> >())
          ("machine.Tanh"         , bpo::value<std::vector<std::string> >())
          ("machine.Softmax"      , bpo::value<std::vector<std::string> >())
          ("machine.SoftmaxStable", bpo::value<std::vector<std::string> >())
          ("machine.SoftmaxClass" , bpo::value<std::vector<std::string> >())
          ("machine.Multi"        , bpo::value<std::vector<std::string> >())
          ("machine.Sequential"   , bpo::value<std::vector<std::string> >())
          ("machine.Parallel"     , bpo::value<std::vector<std::string> >())
          ("machine.Split"        , bpo::value<std::vector<std::string> >())
          ("machine.Split1"       , bpo::value<std::vector<std::string> >())
          ("machine.Join"         , bpo::value<std::vector<std::string> >())
          ("machine.Combined"     , bpo::value<std::vector<std::string> >())
          ("machine.Avr"          , bpo::value<std::vector<std::string> >())
          ("machine.Copy"         , bpo::value<std::vector<std::string> >())
          ;
  this->odGeneralConfig.add(this->odMachineTypes);


  /* set dimension constant names */

  char sDimVar[10];
  for (char c = 1 ; 20 >= c ; c++) {
    sprintf(sDimVar, "DIM%d", c);
    this->odGeneralConfig.add_options()(sDimVar, bpo::value<int>());
  }

  /* set machine specific options */

  // machine options for many machine types except multiple machines
  this->odMachineConf.add_options()
          ("input-dim"            , bpo::value<int> ()->required(), "input dimension")
          ("output-dim"           , bpo::value<int> ()->required(), "output dimension")
          ("nb-forward"           , bpo::value<int> ()->default_value(0), "forward number")
          ("nb-backward"          , bpo::value<int> ()->default_value(0), "backward number")
          ("update"               , bpo::value<bool>(), "update parameters during backward (default true)")
          ("lrate-coeff"          , bpo::value<REAL>(), "layer specific coefficient of the learning rate (default 1.0)")
          ("share-id"          	  , bpo::value<int> ()->default_value(-1), "All machines sharing the same share-id will share their weights (default is all machines share their weights)")
          ;

  // machine options for all machine types (including multiple machines)
  this->odMachMultiConf.add_options()
          ("drop-out"             , bpo::value<REAL>(), "percentage of neurons to be used for drop-out [0-1], set to 0 to turn it off")
          ("block-size"           , bpo::value<int> (), "block size for faster training")
          ("init-from-file"       , bpo::value<std::string>(), "name of file containing all machine data")
          ("name"                 , bpo::value<std::string>(), "name of machine (used internally)")
          ("clone"                , bpo::value<std::string>(), "replace current machine by a copy of previous machine with given name (sharing the parameters)")
          ;
  this->odMachineConf.add(this->odMachMultiConf);
  
  // machine options for multiple machine types ONLY 
  this->odMachMultiConf.add_options()
          ("repeat"             , bpo::value<int>()->default_value(1), "repeat the inner machines N times")
          ;

  // machine options for linear machines (base class MachLin)
  this->odMachLinConf.add_options()
          ("const-init-weights"   , bpo::value<REAL>(), "constant value for initialization of the weights")
          ("ident-init-weights"   , bpo::value<REAL>(), "initialization of the weights by identity transformation")
          ("fani-init-weights"    , bpo::value<REAL>(), "random initialization of the weights by function of fan-in")
          ("fanio-init-weights"   , bpo::value<REAL>(), "random initialization of the weights by function of fan-in and fan-out")
          ("random-init-weights"  , bpo::value<REAL>(), "value for random initialization of the weights (method used by default with general value)")
          ("const-init-bias"      , bpo::value<REAL>(), "constant value for initialization of the bias")
          ("random-init-bias"     , bpo::value<REAL>(), "value for random initialization of the bias (method used by default with general value)")
          ("clip-weights"         , bpo::value<REAL>(), "value for clipping weights (used by default with general value)")
          ("clip-gradients-weights",bpo::value<REAL>(), "value for clipping gradients on weights (used by default with general value)")
          ("clip-gradients-bias"  , bpo::value<REAL>(), "value for clipping gradients on biases (used by default with general value)")
          ;
  this->odMachLinConf.add(this->odMachineConf);

  // machine options for table lookup machines (base class MachTab)
  this->odMachTabConf.add_options()
          ("const-init-project"   , bpo::value<REAL>(), "constant value for initialization of the projection layer")
          ("random-init-project"  , bpo::value<REAL>(), "value for random initialization of the projection layer (method used by default with general value)")
          ;
  this->odMachTabConf.add(this->odMachineConf);


}

/**
 * parses options from command line and configuration file
 * @param iArgCount number of command line arguments
 * @param sArgTable table of command line arguments
 * @return false in case of error or help request, true otherwise
 * @note error code is set if an error occurred
 */
bool MachConfig::parse_options (int iArgCount, char *sArgTable[])
{
  this->vmGeneralOptions.clear();

  // program name
  if (iArgCount > 0) {
    this->sProgName = sArgTable[0];
    size_t stEndPath = this->sProgName.find_last_of("/\\");
    if (stEndPath != std::string::npos)
      this->sProgName.erase(0, stEndPath + 1);
  }
  else
    this->sProgName.clear();

  // set option list used by the application
  bpo::options_description odUsedOptions;
  odUsedOptions.add(this->odCommandLine);
  odUsedOptions.add(this->odSelectedConfig);

  // parse command line
  try {
    bpo::store(bpo::command_line_parser(iArgCount, sArgTable).options(odUsedOptions).positional(this->podCommandLine).run(), this->vmGeneralOptions);

    // verify help option
    this->bHelpRequest = (this->vmGeneralOptions.count("help") > 0);
    if (this->bHelpRequest)
      return false;

    // get configuration file name
    std::vector<std::string> vs;
    std::string sConfFileOpt("config-file");
    if (this->vmGeneralOptions.count(sConfFileOpt) > 0)
      vs = this->vmGeneralOptions[sConfFileOpt].as< std::vector<std::string> >();
    switch (vs.size()) {
    case 1:
      this->sConfFile = vs.front();
      break;
    case 0:
      this->sConfFile.clear();
      if (this->bNeedConfFile) {
        // error: configuration file is required
        throw bpo::required_option(sConfFileOpt);
      }
      else {
        // don't parse configuration file, so notify command line parsing
        bpo::notify(this->vmGeneralOptions);
        return true;
      }
      break;
    default:
      bpo::multiple_occurrences mo;
      mo.set_option_name(sConfFileOpt);
      throw mo;
      break;
    }

  } catch (bpo::error &e) {
    // error handling
    this->eErrorCode = MachConfig::CmdLineParsingError;
    this->ossErrorInfo.str(e.what());
    return false;
  }

  // open configuration file
  if (!this->open_file())
    return false;

  try {
    // parse configuration file and parse command line one more time (to be sure to use selected options with the good attributes)
    bpo::store(bpo::parse_config_file(this->ifsConf, this->odGeneralConfig), this->vmGeneralOptions);
    bpo::store(bpo::command_line_parser(iArgCount, sArgTable).options(odUsedOptions).positional(this->podCommandLine).run(), this->vmGeneralOptions);
    bpo::notify(this->vmGeneralOptions);
  } catch (bpo::error &e) {
    // error handling
    this->eErrorCode = MachConfig::ConfigParsingError;
    this->ossErrorInfo.str(e.what());
    return false;
  }

  // remove unused information (machine structure which will be read without boost)
  const std::vector<boost::shared_ptr<bpo::option_description> >& vodMachOpt = this->odMachineTypes.options();
  std::vector<boost::shared_ptr<bpo::option_description> >::const_iterator iEnd = vodMachOpt.end();
  for (std::vector<boost::shared_ptr<bpo::option_description> >::const_iterator iO = vodMachOpt.begin() ; iO != iEnd ; iO++) {
    bpo::option_description *pod = iO->get();
    if (pod != NULL)
      this->vmGeneralOptions.erase(pod->long_name());
  }

  return true;
}

/**
 * prints help message on standard output
 */
void MachConfig::print_help () const
{
  std::cout <<
      "Usage: " << this->sProgName << " [options]" << std::endl <<
      "       " << this->sProgName << " configuration_file_name [options]" << std::endl <<
      std::endl << this->odCommandLine << std::endl;
  if (this->bSelectedOptions)
    std::cout << this->odSelectedConfig << std::endl;
}

/**
 * reads machine structure from configuration file
 * @return new machine object, or NULL in case of error
 * @note error code is set if an error occurred
 */
Mach *MachConfig::get_machine ()
{
  // open configuration file
  if (!this->open_file())
    return NULL;

  // search for "machine" group
  std::string sRead;
  char sMachGroup[] = "[machine]";
  do {
    this->ifsConf >> sRead;
    std::ios_base::iostate iost = this->ifsConf.rdstate();
    if (iost) {
      // error handling
      if (iost & std::ios_base::eofbit)
        this->eErrorCode = MachConfig::NoMachineGroup;
      else
        this->eErrorCode = MachConfig::ProbSearchMachGroup;
      return NULL;
    }
  } while (sRead != sMachGroup);

  Mach::SetFileId(file_header_version); //Loic: needed to create old machines with new code

  // read machine structure
  this->bReadMachOnly = false;
  this->eErrorCode = MachConfig::NoError;
  Mach *pNextMach = NULL;
  this->read_next_machine(pNextMach, this->iBlockSize);
  if ((this->eErrorCode != MachConfig::NoError) && (pNextMach != NULL)) {
    delete pNextMach;
    pNextMach = NULL;
  }
  this->mMachNameMap.clear();
  return pNextMach;
}

/**
 * get last error
 * @return error string
 */
std::string MachConfig::get_error_string() const
{
  std::string sError;

  // get string
  switch (this->eErrorCode) {
  case MachConfig::NoError:
    return std::string();
    break;
  case MachConfig::CmdLineParsingError:
    sError = "command line error: ";
    sError += this->ossErrorInfo.str();
    return sError;
    break;
  case MachConfig::ProbOpenConfigFile:
    sError = "can't open configuration file \"";
    sError += this->sConfFile;
    sError += '\"';
    return sError;
    break;
  case MachConfig::ConfigParsingError:
    sError = "configuration error: ";
    sError += this->ossErrorInfo.str();
    return sError;
    break;
  case MachConfig::NoMachineGroup:
    return "no [machine] group in configuration file";
    break;
  case MachConfig::ProbSearchMachGroup:
    return "internal error while searching [machine] group";
    break;
  case MachConfig::MachDescrIncomplete:
    return "machine description is not complete";
    break;
  case MachConfig::ProbReadMachName:
    return "internal error while reading machine type name";
    break;
  case MachConfig::UnknownMachType:
    sError = "unknown machine type \"";
    break;
  case MachConfig::UnknownMachName:
    sError = "unknown machine name \"";
    break;
  case MachConfig::UnknownMachCode:
    sError = "unknown machine code ";
    sError += this->ossErrorInfo.str();
    return sError;
    break;
  case MachConfig::MachWithoutEqualChar:
    sError = "no equal character after machine name in \"";
    break;
  case MachConfig::ProbReadMachParams:
    sError = "internal error while reading machine parameters in \"";
    break;
  case MachConfig::MachParamsParsingError:
    sError = "machine parameters error in \"";
    sError += this->ossErrorInfo.str();
    return sError;
    break;
  case MachConfig::ProbOpenMachineFile:
    sError = "can't open machine data file \"";
    break;
  case MachConfig::ProbAllocMachine:
    sError = "can't allocate machine \"";
    break;
  default:
    std::ostringstream oss;
    oss << "unknown error " << this->eErrorCode;
    return oss.str();
    break;
  };

  // append machine type
  sError += this->ossErrorInfo.str();
  sError += '\"';

  return sError;
}

/**
 * get file name of the machine (or void string if not set)
 * @note if mach option value is "%CONF", file name will be same as configuration file (without extension ".conf") followed by extension ".mach"
 */
std::string MachConfig::get_mach () const
{
  const boost::program_options::variable_value &vvM = this->vmGeneralOptions["mach"];
  if (vvM.empty())
    // mach option not set
    return std::string();
  else {
    const std::string &sMachOpt = vvM.as<std::string>();
    if ((sMachOpt == "%CONF") && !this->sConfFile.empty()) {
      size_t stConfFileLen = this->sConfFile.length();

      std::string sConfExt(".conf");
      size_t stConfExtLen = sConfExt.length();

      // verify config-file extension
      if (    (   stConfFileLen   >=  stConfExtLen    )
          &&  (this->sConfFile.compare(stConfFileLen - stConfExtLen, stConfExtLen, sConfExt) == 0)    )
        stConfFileLen -= stConfExtLen;

      // return mach value as config-file value with new extension
      std::string sMachVal(this->sConfFile, 0, stConfFileLen);
      sMachVal.append(".mach");
      return sMachVal;
    }
    else
      // return mach value as set
      return sMachOpt;
  }
}

#ifdef BLAS_CUDA
/**
 * get CUDA devices
 * @returns list of indexes (eg ":0:2" for devices 0 and 2) or number of devices
 */
std::string MachConfig::get_cuda_devices () const
{
  std::string sCudaDev;
  if (this->vmGeneralOptions.count("cuda-device") > 0) {
    // concatenate all device selections (for backward compatibility)
    std::vector<std::string> vsInput = this->vmGeneralOptions["cuda-device"].as<std::vector<std::string> >();
    for (std::vector<std::string>::const_iterator vsci = vsInput.begin() ; vsci != vsInput.end() ; vsci++)
      (sCudaDev += ':') += *vsci;
  }
  else {
    // get number of devices
    std::ostringstream oss;
    oss << this->vmGeneralOptions["cuda-dev-num"].as<int>();
    sCudaDev = oss.str();
  }
  return sCudaDev;
}
#endif

/**
 * open configuration file
 * @return false in case of error, true otherwise
 */
bool MachConfig::open_file ()
{
  this->ifsConf.close();
  this->ifsConf.clear();

  this->ifsConf.open(this->sConfFile.c_str(), std::ios_base::in);
  if (this->ifsConf.fail()) {
    this->eErrorCode = MachConfig::ProbOpenConfigFile;
    return false;
  }
  else {
    this->ifsConf.clear();
    return true;
  }
}

/**
 * reads next machine block from configuration file
 * @param pNewMach set to new machine object pointer, or NULL if 'end' mark is read (and possibly in case of error)
 * @param iBlockSize block size for faster training
 * @return true if 'end' mark is read, false otherwise
 * @note error code is set if an error occurred
 */
bool MachConfig::read_next_machine (Mach *&pNewMach, int iBlockSize)
{
  // read machine type name
  std::string sMachType;
  const char *sMachType_cstr;
  do {
    this->ossErrorInfo.str(sMachType);
    this->ifsConf >> sMachType;
    std::ios_base::iostate iost = this->ifsConf.rdstate();
    if (iost) {
      // error handling
      if (iost & std::ios_base::eofbit)
        this->eErrorCode = MachConfig::MachDescrIncomplete;
      else
        this->eErrorCode = MachConfig::ProbReadMachName;
      this->ossErrorInfo << sMachType;
      pNewMach = NULL;
      return false;
    }
    sMachType_cstr = sMachType.c_str();

    // discard comments / read 'end' mark
    if ('#' == sMachType_cstr[0]) {
      if (strcasecmp(sMachType_cstr, "#End") == 0) {
        pNewMach = NULL;
        return true;
      }
      else {
        std::stringbuf sb;
        this->ifsConf.get(sb);
        this->ifsConf.clear();
        sMachType_cstr = NULL;
      }
    }
  } while (NULL == sMachType_cstr);

  // verify if name contains equal sign
  size_t stEqualPos = sMachType.find('=', 1);
  if (stEqualPos != std::string::npos) {
    this->ifsConf.seekg(stEqualPos - sMachType.length(), std::ios_base::cur);
    this->ifsConf.clear();
    sMachType.resize(stEqualPos);
  }
  this->ossErrorInfo << sMachType;

  // get machine type
  int iMachType;
  bool bMachLin   = false;
  bool bMachMulti = false;
  bool bMachTab   = false;
  if (strcasecmp(sMachType_cstr, "Mach") == 0) {
    iMachType = file_header_mtype_base;
  }
  else if (strcasecmp(sMachType_cstr, "Tab") == 0) {
    iMachType = file_header_mtype_tab;
    bMachTab = true;
  }
  /*else if (strcasecmp(sMachType_cstr, "Tabsh") == 0) {
    iMachType = file_header_mtype_tabsh;
    bMachTab = true;
  }*/
  else if (strcasecmp(sMachType_cstr, "Linear") == 0) {
    iMachType = file_header_mtype_lin;
    bMachLin = true;
  }
  else if (strcasecmp(sMachType_cstr, "Copy") == 0) {
    iMachType = file_header_mtype_copy;
  }
  else if (strcasecmp(sMachType_cstr, "Sig") == 0) {
    iMachType = file_header_mtype_sig;
    bMachLin = true;
  }
  else if (strcasecmp(sMachType_cstr, "Tanh") == 0) {
    iMachType = file_header_mtype_tanh;
    bMachLin = true;
  }
  else if (strcasecmp(sMachType_cstr, "Softmax") == 0) {
    iMachType = file_header_mtype_softmax;
    bMachLin = true;
  }
  /*else if (strcasecmp(sMachType_cstr, "Stab") == 0) {
    iMachType = file_header_mtype_stab;
    bMachLin = true;
  }*/
  else if (strcasecmp(sMachType_cstr, "SoftmaxClass") == 0) {
    iMachType = file_header_mtype_softmax_class;
    bMachLin = true;
  }
  else if (strcasecmp(sMachType_cstr, "SoftmaxStable") == 0) {
    iMachType = file_header_mtype_softmax_stable;
    bMachLin = true;
  }
  else if (strcasecmp(sMachType_cstr, "LinRectif") == 0) {
    iMachType = file_header_mtype_lin_rectif;
    bMachLin = true;
  }
  else {
    bMachMulti = true;
    if (strcasecmp(sMachType_cstr, "Multi") == 0)
      iMachType = file_header_mtype_multi;
    else if (strcasecmp(sMachType_cstr, "Sequential") == 0)
      iMachType = file_header_mtype_mseq;
    else if (strcasecmp(sMachType_cstr, "Split1") == 0)
      iMachType = file_header_mtype_msplit1;
    else if (strcasecmp(sMachType_cstr, "Parallel") == 0)
      iMachType = file_header_mtype_mpar;
    else if (strcasecmp(sMachType_cstr, "Split") == 0)
      iMachType = file_header_mtype_msplit;
    else if (strcasecmp(sMachType_cstr, "Combined") == 0)
      iMachType = file_header_mtype_combined;
    /*else if (strcasecmp(sMachType_cstr, "Max") == 0)		// under development
      iMachType = file_header_mtype_max;*/			// under development
    else if (strcasecmp(sMachType_cstr, "Avr") == 0)
      iMachType = file_header_mtype_avr;
    /*else if (strcasecmp(sMachType_cstr, "Stacked") == 0)	// under development
      iMachType = file_header_mtype_mstack; */			// under development
    else if (strcasecmp(sMachType_cstr, "Join") == 0)
      iMachType = file_header_mtype_mjoin;
    else {
      // error handling
      this->eErrorCode = MachConfig::UnknownMachType;
      pNewMach = NULL;
      return false;
    }
  }

  // create machine
  if (bMachMulti)
    pNewMach = this->read_multi_machine (iMachType, iBlockSize);
  else
    pNewMach = this->read_simple_machine(iMachType, iBlockSize, bMachLin, bMachTab);
  return false;
}

/**
 * creates a multiple machine, reads his parameters and reads submachine blocks
 * @param iMachType type of multiple machine
 * @param iBlockSize block size for faster training
 * @return new machine object (may be NULL in case of error)
 * @note error code is set if an error occurred
 */
Mach *MachConfig::read_multi_machine (int iMachType, int iBlockSize)
{
  Mach *pNewMach = NULL;
  MachMulti *pMachMulti = NULL;
  bool bNoCloneOrInit = true;

  // read machine parameters
  bpo::variables_map vmMachParams;
  if (!this->read_machine_parameters(this->odMachMultiConf, vmMachParams))
    return NULL;

  // get current block size (get current machine block size if defined, or block size in parameter)
  const boost::program_options::variable_value &vvBS = vmMachParams["block-size"];
  int iCurBlockSize = (vvBS.empty() ? iBlockSize : vvBS.as<int>());
  
  // get current repeat content (get current repeat value if defined)
  const boost::program_options::variable_value &vvRPT = vmMachParams["repeat"];
  int iCurRepeat = (vvRPT.empty() ? iRepeat : vvRPT.as<int>());

  // verify if machine structure must be read without creating new object
  if (!this->bReadMachOnly) {
    if (bNoCloneOrInit) {
      // verify if machine is copied from other one
      const boost::program_options::variable_value &vvC = vmMachParams["clone"];
      if (!vvC.empty()) {
        std::string sOtherName = vvC.as<std::string>();
        if (this->mMachNameMap.count(sOtherName) > 0) {
          pNewMach = this->mMachNameMap[sOtherName]->Clone();
          sOtherName.clear();
        }
        if (pNewMach == NULL) {
          // error handling
          if (sOtherName.empty())
            this->eErrorCode = MachConfig::ProbAllocMachine;
          else {
            this->ossErrorInfo.str(sOtherName);
            this->eErrorCode = MachConfig::UnknownMachName;
          }
          return NULL;
        }
        bNoCloneOrInit = false;
      }
    }
    if (bNoCloneOrInit) {
      // verify if machine is read from a file
      const boost::program_options::variable_value &vvIFF = vmMachParams["init-from-file"];
      if (!vvIFF.empty()) {
        pNewMach = this->read_machine_from_file(vvIFF.as<std::string>(), iCurBlockSize, vmMachParams);
        if (pNewMach == NULL)
          // error handling
          return NULL;
        bNoCloneOrInit = false;
      }
    }
    if (bNoCloneOrInit) {
      // instantiate multi machine corresponding to given type
      switch (iMachType) {
      case file_header_mtype_multi:
        pMachMulti = new MachMulti;
        break;
      case file_header_mtype_mseq:
        pMachMulti = new MachSeq;
        break;
      case file_header_mtype_msplit1:
        pMachMulti = new MachSplit1;
        break;
      case file_header_mtype_mpar:
        pMachMulti = new MachPar;
        break;
      case file_header_mtype_msplit:
        pMachMulti = new MachSplit;
        break;
      case file_header_mtype_combined:
        pMachMulti = new MachCombined;
        break;	
      /*case file_header_mtype_max:	// under development
        pMachMulti = new MachMax;
        break;*/
      case file_header_mtype_avr:
        pMachMulti = new MachAvr;
        break;
      /*case file_header_mtype_mstack:	// under development
        pMachMulti = new MachStacked;
        break; */
      case file_header_mtype_mjoin:
        pMachMulti = new MachJoin;
        break;
      default:
        this->eErrorCode = MachConfig::UnknownMachCode;
        this->ossErrorInfo.str(std::string());
        this->ossErrorInfo << iMachType;
        return NULL;
        break;
      }
      if (pMachMulti == NULL) {
        // error handling
        this->eErrorCode = MachConfig::ProbAllocMachine;
        return NULL;
      }
      pNewMach = pMachMulti;

      // apply drop-out parameter (current machine drop-out value if defined, or general value)
      const boost::program_options::variable_value &vvDO = vmMachParams["drop-out"];
      pNewMach->SetDropOut(vvDO.empty() ? this->rPercDropOut : vvDO.as<REAL>());

      // store name of machine if defined
      const boost::program_options::variable_value &vvN = vmMachParams["name"];
      if (!vvN.empty())
        this->mMachNameMap[vvN.as<std::string>()] = pNewMach;
    }
    else
      this->bReadMachOnly = true;
  }

  // read submachines
#ifdef BLAS_CUDA
  size_t stMachConf = ((pMachMulti != NULL) ? pMachMulti->GetGpuConfig() : 0);
  bool bChangeDev = ((Gpu::GetDeviceCount() > 1) && (pMachMulti != NULL) && (
                          (iMachType == file_header_mtype_msplit)
                       || (iMachType == file_header_mtype_mjoin )
                      ));
#endif
  do {
#ifdef BLAS_CUDA
    if (bChangeDev)
      Gpu::NewConfig();
#endif
    Mach *pSubMach = NULL;
    if (this->read_next_machine(pSubMach, iCurBlockSize))
      break;
    else if (pSubMach != NULL) {
      // handle errors
      if (this->eErrorCode != MachConfig::NoError) {
        delete pSubMach;
        break;
      }

      // add new submachine to multi machine
      if (pMachMulti != NULL) {
        pMachMulti->MachAdd(pSubMach);
#ifdef BLAS_CUDA
        Gpu::SetConfig(pSubMach->GetGpuConfig());
#endif
      }
    }
  } while (this->eErrorCode == MachConfig::NoError);
#ifdef BLAS_CUDA
  Gpu::SetConfig(stMachConf); // reset to multi machine GPU
#endif

    if(iCurRepeat > 1){
	int nb = pMachMulti->MachGetNb(); 
	cout << " - repeating these " << nb << " machine(s) " << iCurRepeat << " times" << endl;
	for(int i=0; i<iCurRepeat-1; ++i){
	    for(int j=0; j<nb; ++j){
		Mach* pClonedMach = pMachMulti->MachGet(j)->Clone();
		pMachMulti->MachAdd(pClonedMach);
	    }
	}
    }

  if (!bNoCloneOrInit)
    this->bReadMachOnly = false;
  return pNewMach;
}

/**
 * creates a simple machine and reads his parameters
 * @param iMachType type of simple machine
 * @param iBlockSize block size for faster training
 * @param bMachLin true if the machine is a linear machine, default false otherwise
 * @param bMachTab true if the machine is a table lookup machine, default false otherwise
 * @return new machine object (may be NULL in case of error)
 * @note error code is set if an error occurred
 */
Mach *MachConfig::read_simple_machine (int iMachType, int iBlockSize, bool bMachLin, bool bMachTab)
{
  Mach *pNewMach = NULL;
  bool bNoCloneOrInit = true;
  int iShareId=-1; 

  // read machine parameters
  bpo::variables_map vmMachParams;
  if (!this->read_machine_parameters (bMachLin ? this->odMachLinConf : (bMachTab ? this->odMachTabConf : this->odMachineConf), vmMachParams))
    return NULL;

  // verify if machine structure must be read without creating new object
  if (this->bReadMachOnly)
    return NULL;

  // get current block size (get current machine block size if defined, or block size in parameter)
  const boost::program_options::variable_value &vvBS = vmMachParams["block-size"];
  int iCurBlockSize = (vvBS.empty() ? iBlockSize : vvBS.as<int>());

  if (bNoCloneOrInit) {
    // verify if machine is copied from other one
    const boost::program_options::variable_value &vvC = vmMachParams["clone"];
    if (!vvC.empty()) {
      std::string sOtherName = vvC.as<std::string>();
      if (this->mMachNameMap.count(sOtherName) > 0) {
        pNewMach = this->mMachNameMap[sOtherName]->Clone();
        sOtherName.clear();
      }
      if (pNewMach == NULL) {
        // error handling
        if (sOtherName.empty())
          this->eErrorCode = MachConfig::ProbAllocMachine;
        else {
          this->ossErrorInfo.str(sOtherName);
          this->eErrorCode = MachConfig::UnknownMachName;
        }
      }
      bNoCloneOrInit = false;
    }
  }
  if (bNoCloneOrInit) {
    // verify if machine is read from a file
    const boost::program_options::variable_value &vvIFF = vmMachParams["init-from-file"];
    if (!vvIFF.empty()) {
      pNewMach = this->read_machine_from_file(vvIFF.as<std::string>(), iCurBlockSize, vmMachParams);
      bNoCloneOrInit = false;
    }
  }
  if (bNoCloneOrInit) {
    // get dimension values
    int iInputDim  = vmMachParams[ "input-dim"].as<int>();
    int iOutputDim = vmMachParams["output-dim"].as<int>();

    // get forward and backward numbers
    int iNbForward  = vmMachParams["nb-forward" ].as<int>();
    int iNbBackward = vmMachParams["nb-backward"].as<int>();

    bool bNewShareId = false; // apply general parameters only if machine with new share-id or no-share (-1)
    // instantiate simple machine corresponding to given type
    MachLin *pMachLin = NULL;
    MachCopy *pMachCopy = NULL;
    MachTab *pMachTab = NULL;

    iShareId = vmMachParams["share-id"].as<int>();
    if(iShareId != -1 && prSharedMachines[iShareId] != NULL) {
	//TODO: should we check the machine type also?
	if(prSharedMachines[iShareId]->GetMType() != iMachType){
	  cerr << "WARNING: machines sharing weights have not the same type, check the config file!" << endl;
	}
	if(iMachType == file_header_mtype_tab){
	    if (prSharedMachines[iShareId]->GetIdim()!=1 || iOutputDim != prSharedMachines[iShareId]->GetOdim()){
		Error("MachTab sharing weights have not the same input/output size, check the config file!");
	    }
	}
	else if(iInputDim != prSharedMachines[iShareId]->GetIdim() || iOutputDim != prSharedMachines[iShareId]->GetOdim()){
	    cerr << "mach[" << iShareId << "]->idim=" << prSharedMachines[iShareId]->GetIdim() << " idim=" << iInputDim << endl;
	    cerr << "mach[" << iShareId << "]->odim=" << prSharedMachines[iShareId]->GetOdim() << " odim=" << iOutputDim << endl;
	  Error("Machines sharing weights have not the same input/output size, check the config file!");
	}
	cout << "Cloning previous machine with share-id " << iShareId << endl;
	pNewMach = prSharedMachines[iShareId]->Clone();
	if(iMachType == file_header_mtype_lin) pMachLin = (MachLin*) pNewMach; 
	else if(iMachType == file_header_mtype_tab) pMachTab = (MachTab*) pNewMach; 
    } else if(iShareId == -1 && prSharedMachines[iShareId] != NULL && iMachType == file_header_mtype_tab) {
	    // special case for MachTab
	    // All MachTab share their weights by default. This is for compatibility with previously built system
	    //  cout << "Create MachTab with share-id " << iShareId << " -> cloning existing machine with that share-id" << endl;
	    if(iInputDim != prSharedMachines[iShareId]->GetIdim() || iOutputDim != prSharedMachines[iShareId]->GetOdim()){
	      Error("Machines sharing weights have not the same input/output size, check the config file!");
	    }
	    pNewMach = pMachTab = ((MachTab*)prSharedMachines[iShareId])->Clone();
    } else {
	if(iShareId==-1) cout << "Creating new machine with no share-id" << endl;
	else cout << "Creating new machine with share-id " << iShareId << endl;
	switch (iMachType) {
	case file_header_mtype_base:
	  pNewMach = new Mach(iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward);
	  break;
	case file_header_mtype_tab:
	    pNewMach = pMachTab = new MachTab(iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward, iShareId);
	  break;
	case file_header_mtype_lin:
	  pNewMach = pMachLin = new MachLin(iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward, iShareId);
	  break;
	case file_header_mtype_copy:
	  pNewMach = pMachCopy = new MachCopy(iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward);
	  break;
	case file_header_mtype_lin_rectif:
	  pNewMach = pMachLin = new MachLinRectif(iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward, iShareId);
	  break;
	case file_header_mtype_sig:
	  pNewMach = pMachLin = new MachSig(iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward, iShareId);
	  break;
	case file_header_mtype_tanh:
	  pNewMach = pMachLin = new MachTanh(iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward, iShareId);
	  break;
	case file_header_mtype_softmax:
	  pNewMach = pMachLin = new MachSoftmax(iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward, iShareId);
	  break;
	/*case file_header_mtype_stab:
	  pNewMach = pMachLin = MachStab(iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward);
	  break;*/
	case file_header_mtype_softmax_stable:
	  pNewMach = pMachLin = new MachSoftmaxStable(iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward, iShareId);
	  break;
	case file_header_mtype_softmax_class:
	  pNewMach = pMachLin = new MachSoftmaxClass(iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward, iShareId);
	  break;
	default:
	  this->eErrorCode = MachConfig::UnknownMachCode;
	  this->ossErrorInfo.str(std::string());
	  this->ossErrorInfo << iMachType;
	  return NULL;
	  break;
	}
	if(iShareId != -1){
	    prSharedMachines[iShareId] = pNewMach;
	}
	bNewShareId = true;
    }

    if (pNewMach == NULL) {
      // error handling
      this->eErrorCode = MachConfig::ProbAllocMachine;
      return NULL;
    }

    // apply update parameter if defined
    const boost::program_options::variable_value &vvU = vmMachParams["update"];
    if (!vvU.empty())
      pNewMach->SetUpdataParams(vvU.as<bool>());
  
    // apply lrate-coeff parameter if defined
    const boost::program_options::variable_value &vvLRC = vmMachParams["lrate-coeff"];
    if (!vvLRC.empty())
      pNewMach->SetLrateCoeff(vvLRC.as<REAL>());

    // apply drop-out parameter (current machine drop-out value if defined, or general value)
    const boost::program_options::variable_value &vvDO = vmMachParams["drop-out"];
    pNewMach->SetDropOut(vvDO.empty() ? this->rPercDropOut : vvDO.as<REAL>());

    // store name of machine if defined
    const boost::program_options::variable_value &vvN = vmMachParams["name"];
    if (!vvN.empty())
      this->mMachNameMap[vvN.as<std::string>()] = pNewMach;

    // initialize MachLin
    if (pMachLin != NULL)
      this->apply_machine_parameters(pMachLin, vmMachParams, bNewShareId);

    // initialize MachTab
    if (pMachTab != NULL)
      this->apply_machine_parameters(pMachTab, vmMachParams, bNewShareId);
  }

  return pNewMach;
}

/**
 * reads machine parameters and fills it in given map
 * @param odMachineConf available options for the machine
 * @param vmMachParams map filled with parameters read
 * @return false in case of error, true otherwise
 */
bool MachConfig::read_machine_parameters (const bpo::options_description &odMachineConf, bpo::variables_map &vmMachParams)
{
  // read equal character
  char cEqual = ' ';
  this->ifsConf >> cEqual;
  bool bNoEqualChar = (cEqual != '=');

  // read until end of line
  std::stringbuf sbParamsLine;
  this->ifsConf.get(sbParamsLine);

  // handle errors
  if (this->ifsConf.bad() || bNoEqualChar) {
    if (bNoEqualChar)
      this->eErrorCode = MachConfig::MachWithoutEqualChar;
    else
      this->eErrorCode = MachConfig::ProbReadMachParams;
    this->ossErrorInfo << ' ' << cEqual << sbParamsLine.str();
    return false;
  }
  this->ifsConf.clear();

  // read abbreviated dimensions (ex: " 128 X 256 ", "DIM0xDIM1")
  std::istringstream issParamsLine(sbParamsLine.str());
  std::vector<std::string> vDims;
  vDims.resize(2);
  issParamsLine >> vDims[0];
  std::size_t stPos = vDims[0].find_first_of("xX");
  char cCross;
  if (std::string::npos == stPos)
    issParamsLine >> cCross >> vDims[1];
  else {
    cCross = vDims[0][stPos++];
    if ('\0' == vDims[0][stPos])
      issParamsLine >> vDims[1];
    else
      vDims[1] = vDims[0].substr(stPos);
    vDims[0].erase(stPos - 1);
  }

  // replace dimension constants by their values
  for (std::vector<std::string>::iterator it = vDims.begin() ; it != vDims.end() ; it++) {
    const boost::program_options::variable_value &vv = this->vmGeneralOptions[*it];
    if (!vv.empty())
      try {
        std::ostringstream oss;
        oss << vv.as<int>();
        (*it) = oss.str();
      } catch (boost::bad_any_cast&) {}
  }

  // verify dimensions
  bpo::parsed_options poDims(&odMachineConf);
  if ((!issParamsLine.fail()) && (('x' == cCross) || ('X' == cCross))) {
    // dimensions available
    poDims.options.push_back(bpo::option(std::string( "input-dim"), std::vector<std::string>(1, vDims[0])));
    poDims.options.push_back(bpo::option(std::string("output-dim"), std::vector<std::string>(1, vDims[1])));
  }
  else {
    // no abbreviated dimensions
    issParamsLine.clear();
    issParamsLine.seekg(0);
  }

  // read other machine parameters
  try {
    std::stringbuf sbOtherParams;
    issParamsLine.get(sbOtherParams);
    bpo::store(poDims, vmMachParams);
    bpo::store(
        bpo::command_line_parser(std::vector<std::string>(1, sbOtherParams.str())).
        extra_style_parser(MachConfig::parse_mach_params).options(odMachineConf).run(), vmMachParams);
    bpo::notify(vmMachParams);
  }
  catch (bpo::error &e) {
    // error handling
    this->eErrorCode = MachConfig::MachParamsParsingError;
    this->ossErrorInfo << " =" << sbParamsLine.str() << "\": " << e.what();
    return false;
  }

  return true;
}

/**
 * parses machine parameters
 * @param vsTokens vector of tokens
 * @return vector of options
 * @note throws exception of class boost::program_options::error in case of error
 */
std::vector<bpo::option> MachConfig::parse_mach_params (const std::vector<std::string> &vsTokens)
{
  std::vector<bpo::option> voParsed;

  // put tokens in stream
  std::stringstream ssTokens;
  std::vector<std::string>::const_iterator iEnd = vsTokens.end();
  for (std::vector<std::string>::const_iterator iT = vsTokens.begin() ; iT != iEnd ; iT++)
    ssTokens << *iT << ' ';

  // read parameters
  ParseParametersLine(ssTokens, voParsed);

  // handle errors
  if (ssTokens.bad())
    throw bpo::error("internal stream error");

  return voParsed;
}

/**
 * creates a machine by reading his data from file
 * @param sFileName machine file name
 * @param iBlockSize block size for faster training
 * @param vmMachParams map of parameters read
 * @return new machine object or NULL in case of error
 * @note error code is set if an error occurred
 */
Mach *MachConfig::read_machine_from_file(const std::string &sFileName, int iBlockSize, const bpo::variables_map &vmMachParams)
{
  std::ifstream ifs;
  this->ossErrorInfo.str(sFileName);

  // open file
  ifs.open(sFileName.c_str(), std::ios_base::in);
  if (ifs.fail()) {
    // error handling
    this->eErrorCode = MachConfig::ProbOpenMachineFile;
    return NULL;
  }

  // read file
  Mach *pNewMach = Mach::Read(ifs, iBlockSize);
  if (pNewMach == NULL) {
    // error handling
    this->eErrorCode = MachConfig::ProbAllocMachine;
    return NULL;
  }

  // apply machine forward and backward parameters (set to 0 if not defined)
  const boost::program_options::variable_value &vvNF = vmMachParams["nb-forward" ];
  const boost::program_options::variable_value &vvNB = vmMachParams["nb-backward"];
  pNewMach->SetNbEx(vvNF.empty() ? 0 : vvNF.as<int>(),
                    vvNB.empty() ? 0 : vvNB.as<int>()   );

  // apply update parameter if defined
  const boost::program_options::variable_value &vvU = vmMachParams["update"];
  if (!vvU.empty())
    pNewMach->SetUpdataParams(vvU.as<bool>());

  // apply machine drop-out parameter if defined
  const boost::program_options::variable_value &vvDO = vmMachParams["drop-out"];
  if (!vvDO.empty())
    pNewMach->SetDropOut(vvDO.as<REAL>());

  // initialize MachLin
  MachLin *pMachLin = dynamic_cast<MachLin *>(pNewMach);
  if (pMachLin != NULL) {
    this->apply_machine_parameters(pMachLin, vmMachParams);
    return pNewMach;
  }

  // initialize MachTab
  MachTab *pMachTab = dynamic_cast<MachTab *>(pNewMach);
  if (pMachTab != NULL) {
    this->apply_machine_parameters(pMachTab, vmMachParams);
    return pNewMach;
  }

  return pNewMach;
}

/**
 * applies parameters to given linear machine
 * @note block size parameter is not applied here
 * @param pMachLin pointer to linear machine object
 * @param vmMachParams map of parameters
 * @param bApplyGenVal true to apply general values to parameters as needed, default false otherwise
 */
void MachConfig::apply_machine_parameters(MachLin *pMachLin, const bpo::variables_map &vmMachParams, bool bApplyGenVal) const
{
  if (pMachLin != NULL) {
    bool bWeigthsNotInit = bApplyGenVal;
    bool bBiasNotInit    = bApplyGenVal;

    // constant value for initialization of the weights
    const boost::program_options::variable_value &vvCIW = vmMachParams["const-init-weights"];
    if (!vvCIW.empty()) {
      pMachLin->WeightsConst(vvCIW.as<REAL>());
      bWeigthsNotInit = false;
    }

    // initialization of the weights by identity transformation
    const boost::program_options::variable_value &vvIIW = vmMachParams["ident-init-weights"];
    if (!vvIIW.empty()) {
      pMachLin->WeightsID(vvIIW.as<REAL>());
      bWeigthsNotInit = false;
    }

    // random initialization of the weights by function of fan-in
    const boost::program_options::variable_value &vvFIIW = vmMachParams["fani-init-weights"];
    if (!vvFIIW.empty()) {
      pMachLin->WeightsRandomFanI(vvFIIW.as<REAL>());
      bWeigthsNotInit = false;
    }

    // random initialization of the weights by function of fan-in and fan-out
    const boost::program_options::variable_value &vvFIOIW = vmMachParams["fanio-init-weights"];
    if (!vvFIOIW.empty()) {
      pMachLin->WeightsRandomFanIO(vvFIOIW.as<REAL>());
      bWeigthsNotInit = false;
    }

    // value for random initialization of the weights
    const boost::program_options::variable_value &vvRIW = vmMachParams["random-init-weights"];
    bool bCurRandInitWeights = !vvRIW.empty();
    if (bCurRandInitWeights || bWeigthsNotInit) { // if no init-weights option is used, a general value is applied
      pMachLin->WeightsRandom(bCurRandInitWeights ? vvRIW.as<REAL>() : this->rInitWeights);
    }

    // constant value for initialization of the bias
    const boost::program_options::variable_value &vvCIB = vmMachParams["const-init-bias"];
    if (!vvCIB.empty()) {
      pMachLin->BiasConst(vvCIB.as<REAL>());
      bBiasNotInit = false;
    }

    // value for random initialization of the bias
    const boost::program_options::variable_value &vvRIB = vmMachParams["random-init-bias"];
    bool bCurRandInitBias = !vvRIB.empty();
    if (bCurRandInitBias || bBiasNotInit) { // if no init-bias option is used, a general value is applied
      pMachLin->BiasRandom(bCurRandInitBias ? vvRIB.as<REAL>() : this->rInitBias);
    }

    // value for clipping weights
    const boost::program_options::variable_value &vvCW = vmMachParams["clip-weights"];
    bool bCurClipWeights = !vvCW.empty();
    if (bCurClipWeights || bApplyGenVal) { // if the option is not used, the general value is applied
      pMachLin->SetClipW(bCurClipWeights ? vvCW.as<REAL>() : this->rClipWeights);
    }

    // value for clipping gradients on weights
    const boost::program_options::variable_value &vvCGW = vmMachParams["clip-gradients-weights"];
    bool bCurClipGradWeights = !vvCGW.empty();
    if (bCurClipGradWeights || bApplyGenVal) { // if the option is not used, the general value is applied
      pMachLin->SetClipGradW(bCurClipGradWeights ? vvCGW.as<REAL>() : this->rClipGradWeights);
    }

    // value for clipping gradients on biases
    const boost::program_options::variable_value &vvCGB = vmMachParams["clip-gradients-bias"];
    bool bCurClipGradBias = !vvCGB.empty();
    if (bCurClipGradBias || bApplyGenVal) { // if the option is not used, the general value is applied
      pMachLin->SetClipGradB(bCurClipGradBias ? vvCGB.as<REAL>() : this->rClipGradBias);
    }
  }
}

/**
 * applies parameters to given table lookup machine
 * @note block size parameter is not applied here
 * @param pMachTab pointer to table lookup machine object
 * @param vmMachParams map of parameters
 * @param bApplyGenVal true to apply general values to parameters as needed, default false otherwise
 */
void MachConfig::apply_machine_parameters(MachTab *pMachTab, const bpo::variables_map &vmMachParams, bool bApplyGenVal) const
{
  if (pMachTab != NULL) {
    bool bTableNotInit = bApplyGenVal;

    // constant value for initialization of the projection layer
    const boost::program_options::variable_value &vvCIP = vmMachParams["const-init-project"];
    if (!vvCIP.empty()) {
      pMachTab->TableConst(vvCIP.as<REAL>());
      bTableNotInit = false;
    }

    // value for random initialization of the projection layer
    const boost::program_options::variable_value &vvRIP = vmMachParams["random-init-project"];
    bool bCurRandInitProj = !vvRIP.empty();
    if (bCurRandInitProj || bTableNotInit) { // if no init-project option is used, a general value is applied
      pMachTab->TableRandom(bCurRandInitProj ? vvRIP.as<REAL>() : this->rInitProjLayer);
    }
  }
}
