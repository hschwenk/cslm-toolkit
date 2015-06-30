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
 * This is a simple program to perform the training of continuous space LMs
 */

using namespace std;
#include <iostream>
#include <strings.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "Tools.h"
#include "Mach.h"
#include "MachConfig.h"
#include "TrainerNgramSlist.h"
#include "TrainerNgramClass.h"
#include "ErrFctSoftmCrossEntNgram.h"
#include "ErrFctSoftmClassCrossEntNgram.h"
#include "Lrate.h"

void usage (MachConfig &mc, bool do_exit=true)
{
   cout <<  endl
        << "cslm_train " << cslm_version << " - a tool to train continuous space language models" << endl
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
  string mach_fname, train_fname, dev_fname, lm_fname, lrate_params;
  int last_it = 0, curr_it = 0;
  int MachSeed=0; // default: don't use
  REAL wdecay = 0;
  bool use_word_class = false;
  Mach *mlp;

  // select available options
  mach_config
    .sel_cmdline_option<std::string>      ("mach,m"               , true )
    .sel_cmdline_option<std::string>      ("train-data,t"         , true )
    .sel_cmdline_option<std::string>      ("dev-data,d"           , false)
    .sel_cmdline_option<std::string>      ("lm,l"                 , false)
    .sel_cmdline_option<std::string>      ("lrate,L"              , false)
    .sel_cmdline_option<REAL>             ("weight-decay,W"       , false)
    .sel_cmdline_option<REAL>             ("drop-out,O"           , false)
    .sel_cmdline_option<int>              ("curr-iter,C"          , false)
    .sel_cmdline_option<int>              ("last-iter,I"          , false)
    .sel_cmdline_option<REAL>             ("random-init-project,r", false)
    .sel_cmdline_option<REAL>             ("random-init-weights,R", false)
    .sel_cmdline_option<REAL>             ("clip-weights,w"       , false)
    .sel_cmdline_option<REAL>             ("clip-gradients-weights,g",false)
    .sel_cmdline_option<REAL>             ("clip-gradients-bias,G", false)
    .sel_cmdline_option<int>              ("block-size,B"         , false)
    .sel_cmdline_option<bool>             ("use-word-class,u"     , false)
#ifdef BLAS_CUDA
    .sel_cmdline_option<std::vector<std::string> >("cuda-device,D", false)
    .sel_cmdline_option<int>              ("cuda-dev-num,N"       , false)
#endif
    ;

  // parse parameters
  if (mach_config.parse_options(argc, argv)) {
    // get parameters
    mach_fname  = mach_config.get_mach();
    lm_fname    = mach_config.get_lm();
    train_fname = mach_config.get_train_data();
    dev_fname   = mach_config.get_dev_data();
    wdecay      = mach_config.get_weight_decay();
    last_it     = mach_config.get_last_iter();
    curr_it     = mach_config.get_curr_iter();
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

  // fix the seed for machines random numbers
  MachSeed = mach_config.get_MachSeed();
  
  
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
    cout << "Creating a new machine" << endl;
    if (MachSeed>0) {
      srand48(MachSeed);
      cout<<  " - initializing with seed "<<MachSeed<<endl;  	    
    }
    cout << " - initializing projections with random values in the range " << mach_config.get_random_init_project() << endl;
    cout << " - initializing weights with random values in the range " << mach_config.get_random_init_weights() << endl;
    cout << " - initializing bias with random values in the range " << mach_config.get_random_init_bias() << endl;

    mlp=mach_config.get_machine();
    if (mlp == NULL)
      Error(mach_config.get_error_string().c_str());
  }

  mlp->Info();

  Trainer *trainer = NULL;
  ErrFct *perrfct = NULL;
  const char * train_fname_cstr = train_fname.c_str();
  const char * dev_fname_cstr = (dev_fname.empty() ? NULL : dev_fname.c_str());

  if (use_word_class) {
    
    perrfct = new ErrFctSoftmClassCrossEntNgram(*mlp);
    if (lm_fname.empty()) {
      trainer = new TrainerNgramClass(mlp, lrate, perrfct,
          train_fname_cstr, dev_fname_cstr,
          wdecay, last_it, curr_it);
    }
    else
      Error("TrainerNgramClassSlist is not implemented. You can use either a language model or word classes, not both.");
  } else {
      cout << " - creating ErrorFunction as SoftmCrossEntNgram" << endl;
    perrfct = new ErrFctSoftmCrossEntNgram(*mlp);
    if (lm_fname.empty()) {
      cout << " - creating Trainer as TrainerNgram" << endl;
      trainer = new TrainerNgram     (mlp, lrate, perrfct,
          train_fname_cstr, dev_fname_cstr,
          wdecay, last_it, curr_it);
    }
    else {
      cout << " - creating Trainer as TrainerNgramSlist" << endl;
      trainer = new TrainerNgramSlist(mlp, lrate, perrfct,
          train_fname_cstr, dev_fname_cstr, lm_fname.c_str(),
          wdecay, last_it, curr_it);
    }
  }
  //cout << "Initial perplexity: " << trainer.TestDev() << endl;

  trainer->TrainAndTest(mach_fname_cstr);

  GpuUnlock();
  if (lrate) delete lrate;
  if (mlp) delete mlp;
  if (trainer) delete trainer;
  if (perrfct) delete perrfct;

  return 0;
}
