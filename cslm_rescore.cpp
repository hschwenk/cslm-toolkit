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
 */

using namespace std;
#include <iostream>
#include <fstream>

#include "NbestCSLM.h"

#include "Mach.h"
#include "MachConfig.h"

void usage (MachConfig &mc, bool do_exit=true)
{
   cout <<  "cslm_rescore " << cslm_version << " - A tool to rescore n-grams" << endl
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
  string in_fname, out_fname, lm_fname, cslm_fname, wl_fname, aux_fname;
  int buf_size = 0, aux_dim = 0;

  // select available options
  mach_config
    .sel_cmdline_option<std::string>("input-file,i" , true )
    .sel_cmdline_option<std::string>("output-file,o", true )
#ifdef LM_SRI
    .sel_cmdline_option<std::string>("lm,l"         , true , "rescore with a SRILM")
    .sel_cmdline_option<int>        ("order"        , false, "order of the SRILM")
#endif
#ifdef LM_KEN
    .sel_cmdline_option<std::string>("lm,l"          , true, "rescore with a KENLM")
    .sel_cmdline_option<int>        ("order"         , false, "order of the KENLM")
#endif
    .sel_cmdline_option<int>        ("buf-size,b"   , false)
    .sel_cmdline_option<std::string>("cslm"         , true )
    .sel_cmdline_option<std::string>("vocab,v"      , true )
    .sel_cmdline_option<std::string>("aux-file,a"   , false)
    .sel_cmdline_option<int>        ("aux-dim,n"    , false)
    ;

  // parse parameters
  if (mach_config.parse_options(argc, argv)) {
    // get parameters
    in_fname   = mach_config.get_input_file();
    out_fname  = mach_config.get_output_file();
    lm_fname   = mach_config.get_lm();
    buf_size   = mach_config.get_buf_size();
    cslm_fname = mach_config.get_cslm();
    wl_fname   = mach_config.get_vocab();
    aux_fname  = mach_config.get_aux_file();
    aux_dim    = (aux_fname.empty() ? 0 : mach_config.get_aux_dim());
  }
  else if (mach_config.help_request())
    usage(mach_config);
  else {
    if (mach_config.parsing_error())
      usage(mach_config, false);
    Error(mach_config.get_error_string().c_str());
  }

  cout <<  "cslm_rescore " << cslm_version << " - A tool to process Moses n-best lists" << endl
       << "Copyright (C) 2015 Holger Schwenk, University of Le Mans, France" << endl << endl;

  cout << " - reading input from file '" << in_fname << "'" << endl;
  inputfilestream inpf(in_fname);
  if (0 < aux_dim)
    cout << " - reading auxiliary data from file '" << aux_fname << "'" << endl;
  inputfilestream auxf(aux_fname);
  cout << " - writing output to file '" << out_fname << "'" << endl;
  outputfilestream outf(out_fname);
 
    // shall we rescore with an CSLM ?
  NbestCSLM cslm;
    cout << " - rescoring with CSLM " << cslm_fname;
    cout << endl;
    cslm.Read((char *)cslm_fname.c_str(), (char *)wl_fname.c_str(), (char *)lm_fname.c_str(), -1, aux_dim);

  vector<string> ngrams;
  REAL *probs = new REAL[buf_size];
  REAL *aux_data = NULL;
  if (0 < aux_dim)
    aux_data = new REAL[aux_dim];

   // main loop
  cout << "\nSTART, probas at" << probs << endl;
  time_t t_beg, t_end;
  time(&t_beg);
  int nb_ngrams=0;
  while (!inpf.eof()) {
    string line;
    getline(inpf,line);
    if (inpf.eof()) break;

    if (0 < aux_dim) {
      for (int i = 0 ; aux_dim > i ; i++)
        auxf >> aux_data[i];
      if (!auxf)
        Error("Not enough auxiliary data available");
    }

    ngrams.push_back(line);
    nb_ngrams++;
    if (nb_ngrams%1000==0) printf(" - %d n-grams processed\r",nb_ngrams); fflush(stdout);

    if (ngrams.size() == (size_t) buf_size) {
      cslm.RescoreNgrams(ngrams, probs, aux_data);
      for (size_t i=0; i<ngrams.size(); i++) 
        outf << ngrams[i] << " ||| " << probs[i] << endl;
      ngrams.clear();
    }
  }

    // perform remaining requests
  if (ngrams.size() > 0) {
    cslm.RescoreNgrams(ngrams, probs);
    for (size_t i=0; i<ngrams.size(); i++) 
      outf << ngrams[i] << " ||| " << probs[i] << endl;
  }

  inpf.close();
  outf.close();
  time(&t_end);
  time_t dur=t_end-t_beg;

    // display final statistics
  printf(" - %d n-grams processed\n",nb_ngrams);
  cslm.Stats();
  cout << " - total time: " << dur/60 << "m" << dur%60 << "s" << endl;

  GpuUnlock();
  if (probs) free(probs);
  if (aux_data) free(aux_data);
  
  return 0;
}
