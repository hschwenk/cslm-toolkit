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

#ifndef _MachConfig_h
#define _MachConfig_h

#include <boost/program_options/option.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/variables_map.hpp>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include "MachMulti.h"
#include "MachLin.h"
#include "MachTab.h"



template<class T>
class opt_sem;

/**
 * reads machine configuration from command line and configuration file
 * @note a configuration file contains miscellaneous parameters and a group "[machine]" which describes machine structure
 */
class MachConfig
{
public:

  /**
   * creates a machine configuration reader
   * @param bNeedConfFile true if configuration file is required on command line, false otherwise
   * @param rInitBias general value for random initialization of the bias (0.1 by default)
   */
  MachConfig (bool bNeedConfFile, REAL rInitBias = 0.1);

  /**
   * selects a general option which can be used in command line
   * @param sName long name of the option eventually followed by a comma and the letter used as shortcut ("long_name" or "long_name,s")
   * @param bRequired true if the option value must occur, false otherwise
   * @param sDescription explanation of the option, or default NULL to use default explanation
   * @return reference to '*this' object
   * @note if given option name is not found or if type T is not the same as option type, a new option will still be created
   */
  template<class T>
  inline MachConfig& sel_cmdline_option (const char *sName, bool bRequired, const char *sDescription = NULL)
  {
    return this->sel_cmdline_option<T>(sName, bRequired, NULL, std::string(), sDescription);
  }

  /**
   * selects a general option which can be used in command line
   * @param sName long name of the option eventually followed by a comma and the letter used as shortcut ("long_name" or "long_name,s")
   * @param tDefaultValue default value which will be used if none is explicitly specified (the type 'T' should provide operator<< for std::ostream)
   * @param sDescription explanation of the option, or default NULL to use default explanation
   * @return reference to '*this' object
   * @note if given option name is not found or if type T is not the same as option type, a new option will still be created
   */
  template<class T>
  inline MachConfig& sel_cmdline_option_def (const char *sName, const T &tDefaultValue, const char *sDescription = NULL)
  {
    std::ostringstream oss;
    oss << tDefaultValue;
    return this->sel_cmdline_option<T>(sName, false, &tDefaultValue, oss.str(), sDescription);
  }

  /**
   * parses options from command line and configuration file
   * @param iArgCount number of command line arguments
   * @param sArgTable table of command line arguments
   * @return false in case of error or help request, true otherwise
   * @note error code is set if an error occurred
   */
  bool parse_options (int iArgCount, char *sArgTable[]);

  /**
   * verifies if user requests for help
   * @return true if help is requested
   */
  inline bool help_request () const { return this->bHelpRequest; }

  /**
   * prints help message on standard output
   */
  void print_help () const;

  /**
   * checks if a parsing error occurred in command line or configuration file (general options)
   * @return true if a parsing error occurred
   */
  inline bool parsing_error () const { return ((this->eErrorCode == MachConfig::CmdLineParsingError) || (this->eErrorCode == MachConfig::ConfigParsingError)); }

  /**
   * reads machine structure from configuration file
   * @return new machine object, or NULL in case of error
   * @note error code is set if an error occurred
   */
  Mach *get_machine ();

  /**
   * get last error
   * @return error string
   */
  std::string get_error_string() const;

  /**
   * get file name of the machine (or void string if not set)
   * @note if mach option value is "%CONF", file name will be same as configuration file (without extension ".conf") followed by extension ".mach"
   */
  std::string get_mach () const;

  /**
   * get word list of the source vocabulary (or void string if not set)
   */
  inline std::string get_src_word_list () const
  {
    const boost::program_options::variable_value &vv = this->vmGeneralOptions["src-word-list"];
    return (vv.empty() ? std::string() : vv.as<std::string>());
  }

  /**
   * get word list of the vocabulary and counts (or void string if not set)
   */
  inline std::string get_tgt_word_list () const
  {
    const boost::program_options::variable_value &vv = this->vmGeneralOptions["tgt-word-list"];
    return (vv.empty() ? std::string() : vv.as<std::string>());
  }

  /**
   * get word list of the vocabulary and counts (or void string if not set)
   */
  inline std::string get_word_list () const
  {
    const boost::program_options::variable_value &vv = this->vmGeneralOptions["word-list"];
    return (vv.empty() ? std::string() : vv.as<std::string>());
  }

  /**
   * get file name of the input n-best list (or void string if not set)
   */
  inline std::string get_input_file () const
  {
    const boost::program_options::variable_value &vv = this->vmGeneralOptions["input-file"];
    return (vv.empty() ? std::string() : vv.as<std::string>());
  }

 /**
  * get file name of the auxiliary data (or void string if not set)
  */
  inline std::string get_aux_file () const
  {
    const boost::program_options::variable_value &vv = this->vmGeneralOptions["aux-file"];
    return (vv.empty() ? std::string() : vv.as<std::string>());
  }

  /**
   * get file name of the output n-best list (or void string if not set)
   */
  inline std::string get_output_file () const
  {
    const boost::program_options::variable_value &vv = this->vmGeneralOptions["output-file"];
    return (vv.empty() ? std::string() : vv.as<std::string>());
  }

  /**
   * get file name of the file with source sentences (or void string if not set)
   */
  inline std::string get_source_file () const
  {
    const boost::program_options::variable_value &vv = this->vmGeneralOptions["source-file"];
    return (vv.empty() ? std::string() : vv.as<std::string>());
  }

  /**
   * get Moses on-disk phrase table (or void string if not set)
   */
  inline std::string get_phrase_table () const
  {
    const boost::program_options::variable_value &vv = this->vmGeneralOptions["phrase-table"];
    return (vv.empty() ? std::string() : vv.as<std::string>());
  }

  /**
   * get secondary Moses phrase table (or void string if not set)
   */
  inline std::string get_phrase_table2 () const
  {
    const boost::program_options::variable_value &vv = this->vmGeneralOptions["phrase-table2"];
    return (vv.empty() ? std::string() : vv.as<std::string>());
  }

  /**
   * get test data (or void string if not set)
   */
  inline std::string get_test_data () const
  {
    const boost::program_options::variable_value &vv = this->vmGeneralOptions["test-data"];
    return (vv.empty() ? std::string() : vv.as<std::string>());
  }

  /**
   * get training data (or void string if not set)
   */
  inline std::string get_train_data () const
  {
    const boost::program_options::variable_value &vv = this->vmGeneralOptions["train-data"];
    return (vv.empty() ? std::string() : vv.as<std::string>());
  }

  /**
   * get development data (or void string if not set)
   */
  inline std::string get_dev_data () const
  {
    const boost::program_options::variable_value &vv = this->vmGeneralOptions["dev-data"];
    return (vv.empty() ? std::string() : vv.as<std::string>());
  }

  /**
   * get file name of the machine when using short lists (or void string if not set)
   */
  inline std::string get_lm () const
  {
    const boost::program_options::variable_value &vv = this->vmGeneralOptions["lm"];
    return (vv.empty() ? std::string() : vv.as<std::string>());
  }

  /**
   * get file name of written log-probas (or void string if not set)
   */
  inline std::string get_output_probas () const
  {
    const boost::program_options::variable_value &vv = this->vmGeneralOptions["output-probas"];
    return (vv.empty() ? std::string() : vv.as<std::string>());
  }

  /**
   * get CSLM (or void string if not set)
   */
  inline std::string get_cslm () const
  {
    const boost::program_options::variable_value &vv = this->vmGeneralOptions["cslm"];
    return (vv.empty() ? std::string() : vv.as<std::string>());
  }

  /**
   * get word-list to be used with the CSLM (or void string if not set)
   */
  inline std::string get_vocab () const
  {
    const boost::program_options::variable_value &vv = this->vmGeneralOptions["vocab"];
    return (vv.empty() ? std::string() : vv.as<std::string>());
  }

  /**
   * get CSTM (or void string if not set)
   */
  inline std::string get_cstm () const
  {
    const boost::program_options::variable_value &vv = this->vmGeneralOptions["cstm"];
    return (vv.empty() ? std::string() : vv.as<std::string>());
  }

  /**
   * get source word-list to be used with the CSTM (or void string if not set)
   */
  inline std::string get_vocab_source () const
  {
    const boost::program_options::variable_value &vv = this->vmGeneralOptions["vocab-source"];
    return (vv.empty() ? std::string() : vv.as<std::string>());
  }

  /**
   * get target word-list to be used with the CSTM (or void string if not set)
   */
  inline std::string get_vocab_target () const
  {
    const boost::program_options::variable_value &vv = this->vmGeneralOptions["vocab-target"];
    return (vv.empty() ? std::string() : vv.as<std::string>());
  }

  /**
   * get coefficients of the feature functions (or void string if not set)
   */
  inline std::string get_weights () const
  {
    const boost::program_options::variable_value &vv = this->vmGeneralOptions["weights"];
    return (vv.empty() ? std::string() : vv.as<std::string>());
  }

  /**
   * get specification of the TM scores to be used
   */
  inline std::string get_tm_scores () const { return this->vmGeneralOptions["tm-scores"].as<std::string>(); }

  /**
   * get learning rate parameters
   */
  inline std::string get_lrate () const { return this->vmGeneralOptions["lrate"].as<std::string>(); }

  /**
   * * get MachSeed : seed value for weights random init
   */
  inline int get_MachSeed () const { return this->vmGeneralOptions["MachSeed"].as<int>();}

  /**
   * get number of hypothesis to read per n-best
   */
  inline int get_inn () const { return this->vmGeneralOptions["inn"].as<int>(); }

  /**
   * get number of hypothesis to write per n-best
   */
  inline int get_outn () const { return this->vmGeneralOptions["outn"].as<int>(); }

  /**
   * get offset to add to n-best ID
   */
  inline int get_offs () const { return this->vmGeneralOptions["offs"].as<int>(); }

  /**
   * get the dimension of auxiliary data
   */
  inline int get_aux_dim () const { return this->vmGeneralOptions["aux-dim"].as<int>(); }

  /**
   * get number of scores in phrase table
   */
  inline int get_num_scores () const { return this->vmGeneralOptions["num-scores"].as<int>(); }

  /**
   * get input context size
   */
  inline int get_ctxt_in () const { return this->vmGeneralOptions["ctxt-in"].as<int>(); }

  /**
   * get output context size
   */
  inline int get_ctxt_out () const { return this->vmGeneralOptions["ctxt-out"].as<int>(); }

  /**
   * get current iteration when continuing training of a neural network
   */
  inline int get_curr_iter () const { return this->vmGeneralOptions["curr-iter"].as<int>(); }

  /**
   * get last iteration of neural network
   */
  inline int get_last_iter () const { return this->vmGeneralOptions["last-iter"].as<int>(); }

  /**
   * get order of the LM to apply on the test data
   */
  inline int get_order () const { return this->vmGeneralOptions["order"].as<int>(); }

  /**
   * get mode of the data
   */
  inline int get_mode () const { return this->vmGeneralOptions["mode"].as<int>(); }

  /**
   * get position of LM score
   */
  inline int get_lm_pos () const { return this->vmGeneralOptions["lm-pos"].as<int>(); }

  /**
   * get position of the TM scores
   */
  inline int get_tm_pos () const { return this->vmGeneralOptions["tm-pos"].as<int>(); }

  /**
   * get position of target words
   */
  inline int get_tg_pos () const { return this->vmGeneralOptions["target-pos"].as<int>(); }

  /**
   * get buffer size
   */
  inline int get_buf_size () const { return this->vmGeneralOptions["buf-size"].as<int>(); }

  /**
   * get block size for faster training
   */
  inline int get_block_size () const { return this->iBlockSize; }

  /**
   * get percentage of drop-out
   */
  inline REAL get_drop_out () const { return this->rPercDropOut; }

  /**
   * get value for random initialization of the projection layer
   */
  inline REAL get_random_init_project () const { return this->rInitProjLayer; }

  /**
   * get value for random initialization of the weights
   */
  inline REAL get_random_init_weights () const { return this->rInitWeights; }

  /**
   * get value for random initialization of the bias
   */
  inline REAL get_random_init_bias () const { return this->rInitBias; }

  /**
   * get value for clipping weights
   */
  inline REAL get_clip_weights () const { return this->rClipWeights; }

  /**
   * get value for clipping gradients on weights
   */
  inline REAL get_clip_gradients_weights () const { return this->rClipGradWeights; }

  /**
   * get value for clipping gradients on biases
   */
  inline REAL get_clip_gradients_bias () const { return this->rClipGradBias; }

  /**
   * get coefficient of weight decay
   */
  inline REAL get_weight_decay () const { return this->vmGeneralOptions["weight-decay"].as<REAL>(); }

  /**
   * get state of inverse back-ward translation model use
   */
  inline bool get_backward_tm () const { return (this->vmGeneralOptions.count("backward-tm") > 0); }

  /**
   * get state of probabilities renormalization
   */
  inline bool get_renormal () const { return (this->vmGeneralOptions.count("renormal") > 0); }

  /**
   * get state of global scores recalculation
   */
  inline bool get_recalc () const { return (this->vmGeneralOptions.count("recalc") > 0); }

  /**
   * get state of n-best list sorting according to the global scores
   */
  inline bool get_sort () const { return (this->vmGeneralOptions.count("sort") > 0); }

  /**
   * get state of lexically different hypothesis reporting
   */
  inline bool get_lexical () const { return (this->vmGeneralOptions.count("lexical") > 0); }

  /**
   * get state of server mode listening
   */
  inline bool get_server () const { return (this->vmGeneralOptions.count("server") > 0); }

  /**
   * get state of stable sorting
   */
  inline bool get_unstable_sort () const { return (this->vmGeneralOptions.count("unstable-sort") > 0); }

  /**
   * get state of using word classes in the output layer
   */
  inline bool get_use_word_class () const { return (this->vmGeneralOptions.count("use-word-class") > 0); }

  /**
   * get state of using factors 
   */
  inline bool get_use_factors () const { return (this->vmGeneralOptions.count("use-factors") > 0); }

  /**
   * get layer specification to dump activities when processing n-grams
   */
  inline std::string get_layerfile () const
  {
    const boost::program_options::variable_value &vv = this->vmGeneralOptions["dump-activities"];
    return (vv.empty() ? std::string() : vv.as<std::string>());
  }

#ifdef BLAS_CUDA
  /**
   * get CUDA devices
   * @returns list of indexes (eg ":0:2" for devices 0 and 2) or number of devices
   */
  std::string get_cuda_devices () const;
#endif


private:

  /**
   * error code type
   */
  enum ErrorCode {
    NoError = 0,
    CmdLineParsingError,
    ProbOpenConfigFile,
    ConfigParsingError,
    NoMachineGroup,
    ProbSearchMachGroup,
    MachDescrIncomplete,
    ProbReadMachName,
    UnknownMachType,
    UnknownMachName,
    UnknownMachCode,
    MachWithoutEqualChar,
    ProbReadMachParams,
    MachParamsParsingError,
    ProbOpenMachineFile,
    ProbAllocMachine
  };

  bool bSelectedOptions; ///< some options are selected
  bool bHelpRequest;  ///< user requests for help
  bool bNeedConfFile; ///< configuration file is required on command line
  bool bReadMachOnly; ///< read machine structure without creating new object
  int  iBlockSize;    ///< general block size for faster training
  int  iRepeat;    ///< repeat sub-machines 
  REAL rPercDropOut;   ///< general percentage of drop-out
  REAL rInitProjLayer; ///< general value for random initialization of the projection layer
  REAL rInitWeights;   ///< general value for random initialization of the weights
  REAL rInitBias;      ///< general value for random initialization of the bias
  REAL rClipWeights;     ///< general value for clipping weights
  REAL rClipGradWeights; ///< general value for clipping gradients on weights
  REAL rClipGradBias;    ///< general value for clipping gradients on biases
  std::string sProgName; ///< program name
  std::string sConfFile; ///< configuration file name
  std::ifstream ifsConf; ///< configuration file stream
  std::ostringstream ossErrorInfo; ///< error information for get_error_string method
  MachConfig::ErrorCode eErrorCode; ///< error code
  boost::program_options::options_description odCommandLine;    ///< options for command line only
  boost::program_options::options_description odGeneralConfig;  ///< general options for configuration file
  boost::program_options::options_description odSelectedConfig; ///< general options selected for command line
  boost::program_options::options_description odMachineTypes;   ///< available machine type names
  boost::program_options::options_description odMachineConf;    ///< options for a general machine
  boost::program_options::options_description odMachMultiConf;  ///< options for a multi machine
  boost::program_options::options_description odMachLinConf;    ///< options for a linear machine
  boost::program_options::options_description odMachTabConf;    ///< options for a table lookup machine
  boost::program_options::positional_options_description podCommandLine; ///< options without name
  boost::program_options::variables_map vmGeneralOptions; ///< map of general options
  std::map<std::string,Mach*> mMachNameMap; ///< map of machine names
  
  std::map<int, Mach *> prSharedMachines; // to store Mach pointers for sharing using clone() function

  /**
   * open configuration file
   * @return false in case of error, true otherwise
   */
  bool open_file ();

  /**
   * reads next machine block from configuration file
   * @param pNewMach set to new machine object pointer, or NULL if 'end' mark is read (and possibly in case of error)
   * @param iBlockSize block size for faster training
   * @return true if 'end' mark is read, false otherwise
   * @note error code is set if an error occurred
   */
  bool read_next_machine (Mach *&pNewMach, int iBlockSize);

  /**
   * creates a multiple machine, reads his parameters and reads submachine blocks
   * @param iMachType type of multiple machine
   * @param iBlockSize block size for faster training
   * @return new machine object (may be NULL in case of error)
   * @note error code is set if an error occurred
   */
  Mach *read_multi_machine (int iMachType, int iBlockSize);

  /**
   * creates a simple machine and reads his parameters
   * @param iMachType type of simple machine
   * @param iBlockSize block size for faster training
   * @param bMachLin true if the machine is a linear machine, default false otherwise
   * @param bMachTab true if the machine is a table lookup machine, default false otherwise
   * @return new machine object (may be NULL in case of error)
   * @note error code is set if an error occurred
   */
  Mach *read_simple_machine (int iMachType, int iBlockSize, bool bMachLin = false, bool bMachTab = false);

  /**
   * reads machine parameters and fills it in given map
   * @param odMachineConf available options for the machine
   * @param vmMachParams map filled with parameters read
   * @return false in case of error, true otherwise
   */
  bool read_machine_parameters (const boost::program_options::options_description &odMachineConf, boost::program_options::variables_map &vmMachParams);

  /**
   * parses machine parameters
   * @param vsTokens vector of tokens
   * @return vector of options
   * @throw boost::program_options::error object in case of error
   */
  static std::vector<boost::program_options::option> parse_mach_params (const std::vector<std::string> &vsTokens);

  /**
   * creates a machine by reading his data from file
   * @param sFileName machine file name
   * @param iBlockSize block size for faster training
   * @param vmMachParams map of parameters read
   * @return new machine object or NULL in case of error
   * @note error code is set if an error occurred
   */
  Mach *read_machine_from_file(const std::string &sFileName, int iBlockSize, const boost::program_options::variables_map &vmMachParams);

  /**
   * applies parameters to given linear machine
   * @note block size parameter is not applied here
   * @param pMachLin pointer to linear machine object
   * @param vmMachParams map of parameters
   * @param bApplyGenVal true to apply general values to parameters as needed, default false otherwise
   */
  void apply_machine_parameters(MachLin *pMachLin, const boost::program_options::variables_map &vmMachParams, bool bApplyGenVal = false) const;

  /**
   * applies parameters to given table lookup machine
   * @note block size parameter is not applied here
   * @param pMachTab pointer to table lookup machine object
   * @param vmMachParams map of parameters
   * @param bApplyGenVal true to apply general values to parameters as needed, default false otherwise
   */
  void apply_machine_parameters(MachTab *pMachTab, const boost::program_options::variables_map &vmMachParams, bool bApplyGenVal = false) const;

  /**
   * selects a general option which can be used in command line
   * @param sName long name of the option eventually followed by a comma and the letter used as shortcut ("long_name" or "long_name,s")
   * @param bRequired true if the option value must occur, false otherwise
   * @param ptDefaultValue pointer to default value which will be used if none is explicitly specified, or NULL if there is no default value
   * @param sTextualValue textual representation of default value
   * @param sDescription explanation of the option, or default NULL to use default explanation
   * @return reference to '*this' object
   * @note if given option name is not found or if type T is not the same as option type, a new option will still be created
   */
  template<class T>
  MachConfig& sel_cmdline_option (const char *sName, bool bRequired, const T *ptDefaultValue, const std::string &sTextualValue, const char *sDescription = NULL)
  {
    if (sName != NULL) {
      boost::program_options::typed_value<T> *ptvtNewSemantic = NULL;
      const char *sNewDescription = "";

      // search for comma in option name
      const char * sShortPart = sName;
      while(((*sShortPart) != ',') && ((*sShortPart) != '\0')) sShortPart++;

      // get option information
      const boost::program_options::option_description *podOption = this->odGeneralConfig.find_nothrow(std::string(sName, sShortPart - sName), false);
      if (podOption != NULL) {
        // get copy of semantic
        const opt_sem<T> *postSemantic = dynamic_cast<const opt_sem<T>*>(podOption->semantic().get());
        if (postSemantic != NULL)
          ptvtNewSemantic = postSemantic->parent_copy();

        // get description
        if (sDescription == NULL)
          sNewDescription = podOption->description().c_str();
      }

      // create new semantic if none were found
      if (ptvtNewSemantic == NULL)
        ptvtNewSemantic = boost::program_options::value<T>();

      // modify semantic
      if (ptvtNewSemantic != NULL) {
        if (ptDefaultValue != NULL)
          ptvtNewSemantic->default_value(*ptDefaultValue, sTextualValue);
        if (bRequired)
          ptvtNewSemantic->required();
      }

      // add new option to command line options
      this->odSelectedConfig.add_options() (    sName,  ptvtNewSemantic,
          (sDescription != NULL) ? sDescription : sNewDescription   );
      this->bSelectedOptions = true;
    }
    return *this;
  }
};


/**
 * handles semantic of a specific option type
 * (give copy function to boost::program_options::typed_value class)
 * @see boost::program_options::typed_value
 */
template<class T>
class opt_sem : public boost::program_options::typed_value<T>
{
public:
  /**
   * creates new option semantic
   * @see boost::program_options::value(T*)
   * @param ptStoreTo pointer to value which will contain the value when it's known (default NULL)
   * @return pointer to new object (to be destroyed)
   */
  static inline opt_sem<T> *new_sem(T* ptStoreTo = NULL)
  {
    return new opt_sem<T>(ptStoreTo);
  }

  /**
   * constructs option semantic
   * @see boost::program_options::typed_value(T*)
   * @param ptStoreTo pointer to value which will contain the value when it's known (can be NULL)
   */
  opt_sem(T *ptStoreTo) :
    boost::program_options::typed_value<T>(ptStoreTo), ptStoreTo(ptStoreTo),
    bDefaultValue(false), bImplicitValue(false), bNotifier(false),
    bComposing(false), bMultitoken(false), bZeroTokens(false), bRequired(false)
  {}

  /**
   * specifies default value, which will be used if none is explicitly specified
   * @see boost::program_options::typed_value::default_value(const T&)
   * @param tValue default value (the type 'T' should provide operator<< for std::ostream)
   * @return pointer to this object
   */
  opt_sem<T> *default_value(const T &tValue)
  {
    this->tDefaultValue = tValue;
    this->bDefaultValue = true;
    std::ostringstream oss;
    oss << tValue;
    this->sDefaultValueText = oss.str();
    boost::program_options::typed_value<T>::default_value(tValue, this->sDefaultValueText);
    return this;
  }

  /**
   * specifies default value, which will be used if none is explicitly specified
   * @see boost::program_options::typed_value::default_value(const T&,const std::string&)
   * @param tValue default value
   * @param sTextual textual representation of default value
   * @return pointer to this object
   */
  opt_sem<T> *default_value(const T &tValue, const std::string &sTextual)
  {
    this->tDefaultValue = tValue;
    this->bDefaultValue = true;
    this->sDefaultValueText = sTextual;
    boost::program_options::typed_value<T>::default_value(tValue, sTextual);
    return this;
  }

  /**
   * specifies an implicit value, which will be used if the option is given, but without an adjacent value
   * @see boost::program_options::typed_value::implicit_value(const T&)
   * @param tValue implicit value (the type 'T' should provide operator<< for std::ostream)
   * @return pointer to this object
   */
  opt_sem<T> *implicit_value(const T &tValue)
  {
    this->tImplicitValue = tValue;
    this->bImplicitValue = true;
    std::ostringstream oss;
    oss << tValue;
    this->sImplicitValueText = oss.str();
    boost::program_options::typed_value<T>::implicit_value(tValue, this->sImplicitValueText);
    return this;
  }

  /**
   * specifies an implicit value, which will be used if the option is given, but without an adjacent value
   * @see boost::program_options::typed_value::implicit_value(const T&,const std::string&)
   * @param tValue implicit value
   * @param sTextual textual representation of implicit value
   * @return pointer to this object
   */
  opt_sem<T> *implicit_value(const T &tValue, const std::string &sTextual)
  {
    this->tImplicitValue = tValue;
    this->bImplicitValue = true;
    this->sImplicitValueText = sTextual;
    boost::program_options::typed_value<T>::implicit_value(tValue, sTextual);
    return this;
  }

  /**
   * specifies a function to be called when the final value is determined
   * @see boost::program_options::typed_value::notifier(boost::function1<void,const T&>)
   * @param f1vt function called
   * @return pointer to this object
   */
  opt_sem<T> *notifier(boost::function1<void,const T&> f1vt)
  {
    this->f1vtNotifier = f1vt;
    this->bNotifier = true;
    boost::program_options::typed_value<T>::notifier(f1vt);
    return this;
  }

  /**
   * specifies that the value is composing
   * @see boost::program_options::typed_value::composing()
   * @return pointer to this object
   */
  opt_sem<T> *composing()
  {
    this->bComposing = true;
    boost::program_options::typed_value<T>::composing();
    return this;
  }

  /**
   * specifies that the value can span multiple tokens
   * @see boost::program_options::typed_value::multitoken()
   * @return pointer to this object
   */
  opt_sem<T> *multitoken()
  {
    this->bMultitoken = true;
    boost::program_options::typed_value<T>::multitoken();
    return this;
  }

  /**
   * specifies that no tokens may be provided as the value of this option
   * @see boost::program_options::typed_value::zero_tokens()
   * @return pointer to this object
   */
  opt_sem<T> *zero_tokens()
  {
    this->bZeroTokens = true;
    boost::program_options::typed_value<T>::zero_tokens();
    return this;
  }

  /**
   * specifies that the value must occur
   * @see boost::program_options::typed_value::required()
   * @return pointer to this object
   */
  opt_sem<T> *required()
  {
    this->bRequired = true;
    boost::program_options::typed_value<T>::required();
    return this;
  }

  /**
   * copies option semantic
   * @return pointer to new parent object (to be destroyed)
   * @see boost::program_options::value(T*)
   */
  boost::program_options::typed_value<T> *parent_copy() const
  {
    boost::program_options::typed_value<T> *ptvt = boost::program_options::value<T>(this->ptStoreTo);
    if (this->bDefaultValue)
      ptvt->default_value(this->tDefaultValue, this->sDefaultValueText);
    if (this->bImplicitValue)
      ptvt->implicit_value(this->tImplicitValue, this->sImplicitValueText);
    if (this->bNotifier)
      ptvt->notifier(this->f1vtNotifier);
    if (this->bComposing)
      ptvt->composing();
    if (this->bMultitoken)
      ptvt->multitoken();
    if (this->bZeroTokens)
      ptvt->zero_tokens();
    if (this->bRequired)
      ptvt->required();
    return ptvt;
  }

private:
  T* ptStoreTo;
  T tDefaultValue, tImplicitValue;
  std::string sDefaultValueText, sImplicitValueText;
  bool bDefaultValue, bImplicitValue, bNotifier;
  bool bComposing, bMultitoken, bZeroTokens, bRequired;
  boost::function1<void,const T&> f1vtNotifier;
};

#endif
