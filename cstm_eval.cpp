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

#include "Mach.h"
#include "MachConfig.h"
#include "DataPhraseBin.h"
#include "TrainerPhraseSlist.h"
#include "ErrFctSoftmCrossEntNgramMulti.h"

void usage (MachConfig &mc, bool do_exit=true)
{
   cout <<  endl
        << "cstm_eval " << cslm_version << " - a tool to evaluate continuous space translation models" << endl
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
  string mach_fname, test_fname, ptable_fname, prob_fname;

  // select available options
  mach_config
    .sel_cmdline_option<std::string>      ("mach,m"         , true )
    .sel_cmdline_option<std::string>      ("phrase-table"   , false , "classical phrase table")
    .sel_cmdline_option<int>              ("num-scores,n"   , false)
    .sel_cmdline_option<std::string>      ("test-data,t"    , true )
    .sel_cmdline_option<std::string>      ("output-probas"  , false)
    .sel_cmdline_option<int>              ("block-size,B"   , false, "block size for faster evaluation")
#ifdef BLAS_CUDA
    .sel_cmdline_option<std::vector<std::string> >("cuda-device,D", false)
    .sel_cmdline_option<int>              ("cuda-dev-num,N"       , false)
#endif
    ;

  // parse parameters
  if (mach_config.parse_options(argc, argv)) {
    // get parameters
    mach_fname   = mach_config.get_mach();
    ptable_fname = mach_config.get_phrase_table();
    test_fname   = mach_config.get_test_data();
    prob_fname   = mach_config.get_output_probas();
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


  cout << "Evaluating CSTM: " << mach_fname << endl;

    // read network
  ifstream ifs;
  const char *mach_fname_cstr = mach_fname.c_str();
  ifs.open(mach_fname_cstr,ios::binary);
  CHECK_FILE(ifs,mach_fname_cstr);
  Mach *m = Mach::Read(ifs);
  ifs.close();
  m->SetBsize(mach_config.get_block_size());
  m->Info();

    // load data
  Data data(test_fname.c_str());

    // create a trainer for testing only
  ErrFctSoftmCrossEntNgramMulti errfct(*m, data.GetOdim());	// TODO: get this from the machine
  TrainerPhraseSlist trainer(m, &errfct, &data,
      (ptable_fname.empty() ? NULL : (char *)ptable_fname.c_str()),
      mach_config.get_num_scores());
  trainer.TestDev(prob_fname.empty() ? NULL : (char *)prob_fname.c_str());
#ifdef PROFILE
  //cout << "Profiling information:" << endl;
  //m->Info();
#endif
  GpuUnlock();
  exit(1); // brute force exit since we have trouble with memory deallocation

  delete m;
  return 0;
}
