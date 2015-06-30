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
 */

using namespace std;
#include <iostream>

#include "Mach.h"
#include "MachConfig.h"
#include "DataNgramBin.h"
#include "TrainerNgramSlist.h"
#include "TrainerNgramClass.h"
#include "ErrFctSoftmCrossEntNgram.h"
#include "ErrFctSoftmClassCrossEntNgram.h"

void usage (MachConfig &mc, bool do_exit=true)
{
   cout <<  endl
        << "cslm_eval " << cslm_version << " - a tool to evaluate continuous space language models" << endl
	<< "Copyright (C) 2015 Holger Schwenk, University of Le Mans, France" << endl << endl;

#if 0
	<< "This library is free software; you can redistribute it and/or" << endl
	<< "modify it under the terms of the GNU Lesser General Public" << endl
	<< "License as published by the Free Software Foundation; either" << endl
	<< "version 2.1 of the License, or (at your option) any later version." << endl << endl

	<< "This library is distributed in the hope that it will be useful," << endl
	<< "but WITHOUT ANY WARRANTY; without even the implied warranty of" << endl
	<< "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU" << endl
	<< "Lesser General Public License for more details." << endl << endl

	<< "You should have received a copy of the GNU Lesser General Public" << endl
	<< "License along with this library; if not, write to the Free Software" << endl
	<< "Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA" << endl << endl
	<< "***********************************************************************" << endl << endl
	<< "Built on " << __DATE__ << endl << endl;
#endif

  mc.print_help();

  if (do_exit) exit(1);
}


int main (int argc, char *argv[])
{
  MachConfig mach_config(false);
  string mach_fname, test_fname, lm_fname, prob_fname;
  bool use_word_class = false;

  // select available options
  mach_config
    .sel_cmdline_option<std::string>      ("mach,m"       , true )
    .sel_cmdline_option<std::string>      ("test-data,t"  , true )
    .sel_cmdline_option<std::string>      ("lm,l"         , false)
    .sel_cmdline_option<std::string>      ("output-probas", false)
    .sel_cmdline_option<bool>             ("renormal,R"   , false)
    .sel_cmdline_option<int>              ("block-size,B" , false, "block size for faster evaluation")
    .sel_cmdline_option<bool>             ("use-word-class,u"     , false)
#ifdef BLAS_CUDA
    .sel_cmdline_option<std::vector<std::string> >("cuda-device,D", false)
    .sel_cmdline_option<int>              ("cuda-dev-num,N"       , false)
#endif
    ;

  // parse parameters
  if (mach_config.parse_options(argc, argv)) {
    // get parameters
    mach_fname = mach_config.get_mach();
    lm_fname   = mach_config.get_lm();
    test_fname = mach_config.get_test_data();
    prob_fname = mach_config.get_output_probas();
    use_word_class = mach_config.get_use_word_class();
#ifdef BLAS_CUDA
    cuda_user_list = mach_config.get_cuda_devices();
#endif
  }
  else if (mach_config.help_request())
    usage(mach_config);
  else {
    if (mach_config.parsing_error())
      usage(mach_config, false);
    Error(mach_config.get_error_string().c_str());
  }


  cout << "Evaluating CSLM: " << mach_fname << endl;

    // read network
  ifstream ifs;
  const char *mach_fname_cstr = mach_fname.c_str();
  ifs.open(mach_fname_cstr,ios::binary);
  CHECK_FILE(ifs,mach_fname_cstr);
  Mach *m = Mach::Read(ifs, mach_config.get_block_size());
  ifs.close();
  m->Info();
#if 0
  m->SetBsize(1);
  REAL idata[]={1,2,3,4};
  m->SetDataIn(idata);
  m->Forw();
  return 0;
#endif

    // load data
  Data data(test_fname.c_str(), NULL, use_word_class);

    // create a trainer for testing only
  char * prob_fname_cstr = (prob_fname.empty() ? NULL : (char *)prob_fname.c_str());

  if (use_word_class) {
    ErrFctSoftmClassCrossEntNgram errfct(*m);
    if (lm_fname.empty()) {
      TrainerNgramClass trainer(m, &errfct, &data);
      cout << "Evaluating:" << endl;
      trainer.TestDev(prob_fname_cstr);
    }
    else
      Error("TrainerNgramClassSlist is not implemented. You can use either a language model or word classes, not both.");
  }
  else {
    ErrFctSoftmCrossEntNgram errfct(*m);
    if (lm_fname.empty()) {
        {
	  TrainerNgram trainer(m, &errfct, &data);
	  cout << "Evaluating:" << endl;
	  trainer.TestDev(prob_fname_cstr);
	}
    }
    else {
      TrainerNgramSlist trainer(m, &errfct, &data, (char *)lm_fname.c_str());
      if (mach_config.get_renormal()) {
        cout << "Evaluating (renormalized with back-off LM proba-mass):" << endl;
        trainer.TestDevRenorm(prob_fname_cstr);
      }
      else {
        cout << "Evaluating:" << endl;
        trainer.TestDev(prob_fname_cstr);
      }
#ifdef PROFILE
      cout << "Profiling information:" << endl;
      m->Info();
#endif
    }
  }
  GpuUnlock();

  delete m;
  return 0;
}
