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
 * This is a simple program to perform the training of continuous space TMs
 */

using namespace std;
#include <iostream>
#include <strings.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "Blas.h"
#include "Mach.h"
#include "MachConfig.h"
#include "TrainerPhraseSlist.h"
#include "ErrFctSoftmCrossEntNgram.h"
#include "Lrate.h"

void usage (MachConfig &mc, bool do_exit=true)
{
   cout <<  endl
        << "cstm_train - a tool to train continuous space translation models" << endl
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
  MachConfig mach_config(true);
  string mach_fname, train_fname, dev_fname, ptable_fname, src_wl_fname, tgt_wl_fname;
  int curr_it = 0;
  Mach *mlp;

  // select available options
  mach_config
    .sel_cmdline_option<std::string>      ("mach,m"               , true )
    .sel_cmdline_option<std::string>      ("phrase-table"         , false, "classical phrase table")
    .sel_cmdline_option<int>              ("num-scores,n"         , false)
    .sel_cmdline_option<std::string>      ("train-data,t"         , true )
    .sel_cmdline_option<std::string>      ("dev-data,d"           , false)
    .sel_cmdline_option<std::string>      ("lrate,L"              , false)
    .sel_cmdline_option<REAL>             ("weight-decay,W"       , false)
    .sel_cmdline_option<int>              ("curr-iter,C"          , false)
    .sel_cmdline_option<int>              ("last-iter,I"          , false)
    .sel_cmdline_option<REAL>             ("random-init-project,r", false)
    .sel_cmdline_option<REAL>             ("random-init-weights,R", false)
    .sel_cmdline_option<REAL>             ("clip-weights,w"       , false)
    .sel_cmdline_option<REAL>             ("clip-gradients-weights,g",false)
    .sel_cmdline_option<REAL>             ("clip-gradients-bias,G", false)
    .sel_cmdline_option<int>              ("block-size,B"         , false)
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
    train_fname  = mach_config.get_train_data();
    dev_fname    = mach_config.get_dev_data();
    curr_it      = mach_config.get_curr_iter();
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

    // verify mandatory arguments
  if ((!dev_fname.empty()) && ptable_fname.empty()) {
    cout << "WARNING: testing on development data without external phrase table" << endl;
    cout << "         phrases which are not in the short-list will be flagged" << endl;
  }

  // read learning rate parameters
  Lrate *lrate = Lrate::NewLrate(mach_config.get_lrate());

    // Check if existing machine exists
  struct stat stat_struct;
  const char *mach_fname_cstr = mach_fname.c_str();
  if (stat(mach_fname_cstr, &stat_struct)==0) {
      // read existing network
    ifstream ifs;
    ifs.open(mach_fname_cstr,ios::binary);
    CHECK_FILE(ifs,mach_fname_cstr);
    mlp = Mach::Read(ifs);
    ifs.close();
    cout << "Found existing machine with " << mlp->GetNbBackw()
         << " backward passes, continuing training at iteration " << curr_it+1 << endl;
  }
  else {
    cout << " - initializing projections with random values in the range " << mach_config.get_random_init_project() << endl;
    cout << " - initializing weights with random values in the range " << mach_config.get_random_init_weights() << endl;

    mlp=mach_config.get_machine();
    if (mlp == NULL)
      Error(mach_config.get_error_string().c_str());
  }

  mlp->Info();

  ErrFctSoftmCrossEntNgram errfct(*mlp);
  Trainer *trainer;
  if (1) {
    trainer = new TrainerPhraseSlist(mlp, lrate, &errfct, train_fname.c_str(),
        (dev_fname.empty() ? NULL : dev_fname.c_str()),
        (ptable_fname.empty() ? NULL : ptable_fname.c_str()),
        mach_config.get_num_scores(), mach_config.get_weight_decay(),
        mach_config.get_last_iter(), curr_it);
  }
  //cout << "Initial perplexity: " << endl;
  //trainer->TestDev();

  char sfname[16384];
  strcpy(sfname,mach_fname_cstr);
  char *p=strstr(sfname, ".mach");
  if (p) {
    *p=0;
    strcat(sfname, dev_fname.empty() ? ".cur.mach" : ".best.mach");
  }
  trainer->TrainAndTest(sfname);

    // save machine at the end
  cout << " - saving final machine into file '" << mach_fname << "'" << endl;
  ofstream fs;
  fs.open(mach_fname_cstr,ios::binary);
  CHECK_FILE(fs,mach_fname_cstr);
  mlp->Write(fs);
  fs.close();

  GpuUnlock();
  if (lrate) delete lrate;
  if (mlp) delete mlp;
  if (trainer) delete trainer;

  return 0;
}
