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
#include <unistd.h>

#include "NBest.h"
#ifdef LM_SRI
#include "NbestLMSRI.h"
#endif
#ifdef LM_KEN
#include "NbestLMKEN.h"
#endif
#include "NbestCSLM.h"
#include "NbestCSTM.h"
#include "PtableMosesPtree.h"

const int PTABLE_NUM_SCORES=5;

#define DUMMY_MACHINE
#ifdef DUMMY_MACHINE
 #include "Mach.h"
#endif
#include "MachConfig.h"

void usage (MachConfig &mc, bool do_exit=true)
{
   cout <<  "NBest " << cslm_version << " - A tool to process Moses n-best lists" << endl
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


/*******************************************************
 *
 *******************************************************/

int main (int argc, char *argv[])
{
  MachConfig mach_config(false);
  string in_fname, out_fname, src_fname,
       w_fname, pt_fname, pt_fname2, lm_fname, cslm_fname, wl_fname,
       cstm_fname, wl_src_fname, wl_tgt_fname, aux_fname;
  int in_n = 0, out_n = 0, offs = 0, tm_pos = 0, lm_pos = 0, lm_order = 0, tg_pos=-1, aux_dim = 0;
  string tm_scores_specif; int tm_nb_scores = 4;
  bool do_ptable = false, do_lm = false, do_cslm = false, do_cstm = false;
  bool do_calc = false, do_sort = false, do_lexical = false, backward_tm = false;

  // select available options
  mach_config
    .sel_cmdline_option<std::string>("input-file,i"  , true )
    .sel_cmdline_option<int>        ("inn,I"         , false)
    .sel_cmdline_option<std::string>("output-file,o" , true )
    .sel_cmdline_option<int>        ("outn,O"        , false)
    .sel_cmdline_option<std::string>("source-file,S" , false)
    .sel_cmdline_option<int>        ("offs,a"        , false)
    .sel_cmdline_option<std::string>("phrase-table"  , false)
    .sel_cmdline_option<std::string>("phrase-table2" , false)
    .sel_cmdline_option<bool>       ("backward-tm,V" , false)
#ifdef LM_SRI
    .sel_cmdline_option<std::string>("lm,l"          , false, "rescore with a SRILM")
    .sel_cmdline_option<int>        ("order"         , false, "order of the SRILM")
#endif
#ifdef LM_KEN
    .sel_cmdline_option<std::string>("lm,l"          , false, "rescore with a KENLM")
    .sel_cmdline_option<int>        ("order"         , false, "order of the KENLM")
#endif
    .sel_cmdline_option<std::string>("cslm"          , false)
    .sel_cmdline_option<std::string>("vocab,v"       , false)
    .sel_cmdline_option<std::string>("cstm"          , false)
    .sel_cmdline_option<std::string>("vocab-source,b", false)
    .sel_cmdline_option<std::string>("vocab-target,B", false)
    .sel_cmdline_option<int>        ("lm-pos,p"      , false)
    .sel_cmdline_option<int>        ("tm-pos,P"      , false)
    .sel_cmdline_option<int>        ("target-pos,P"  , false)
    .sel_cmdline_option<std::string>("tm-scores,N"   , false)
    .sel_cmdline_option<bool>       ("recalc,r"      , false)
    .sel_cmdline_option<std::string>("weights,w"     , false)
    .sel_cmdline_option<bool>       ("sort,s"        , false)
    .sel_cmdline_option<bool>       ("lexical,h"     , false)
    .sel_cmdline_option<bool>       ("server,X"      , false)
    .sel_cmdline_option<bool>       ("unstable-sort" , false)
    .sel_cmdline_option<std::string>("aux-file"      , false)
    .sel_cmdline_option<int>        ("aux-dim"       , false)
    ;

  // parse parameters
  if (mach_config.parse_options(argc, argv)) {
    // get parameters
    in_fname         = mach_config.get_input_file();
    in_n             = mach_config.get_inn();
    out_fname        = mach_config.get_output_file();
    out_n            = mach_config.get_outn();
    src_fname        = mach_config.get_source_file();
    offs             = mach_config.get_offs();
    pt_fname         = mach_config.get_phrase_table();
    pt_fname2        = mach_config.get_phrase_table2();
    backward_tm      = mach_config.get_backward_tm();
    lm_fname         = mach_config.get_lm();
    lm_order         = mach_config.get_order();
    lm_pos           = mach_config.get_lm_pos();
    tm_pos           = mach_config.get_tm_pos();
    tg_pos           = mach_config.get_tg_pos();
    tm_scores_specif = mach_config.get_tm_scores();
    cslm_fname       = mach_config.get_cslm();
    wl_fname         = mach_config.get_vocab();
    cstm_fname       = mach_config.get_cstm();
    wl_src_fname     = mach_config.get_vocab_source();
    wl_tgt_fname     = mach_config.get_vocab_target();
    w_fname          = mach_config.get_weights();
    aux_fname        = mach_config.get_aux_file();
    aux_dim          = (aux_fname.empty() ? 0 : mach_config.get_aux_dim());
    do_calc          = mach_config.get_recalc();
    do_sort          = mach_config.get_sort();
    do_lexical       = mach_config.get_lexical();
    do_ptable        = !  pt_fname.empty();
    do_lm            = !  lm_fname.empty();
    do_cslm          = !cslm_fname.empty();
    do_cstm          = !cstm_fname.empty();
  }
  else if (mach_config.help_request())
    usage(mach_config);
  else {
    if (mach_config.parsing_error())
      usage(mach_config, false);
    Error(mach_config.get_error_string().c_str());
  }

  if ((do_cstm || do_ptable) && src_fname.empty()) {
    usage(mach_config, false);
    Error("\nthe source file is required when rescoring translation model probabilities");
  }

  if ((!pt_fname2.empty()) && !do_ptable) {
    usage(mach_config, false);
    Error("\na primary phrase table is needed");
  }

  if (do_cstm && (wl_src_fname.empty() || wl_tgt_fname.empty()))
    Error("source and target word lists are required when rescoring with a CSTM");

  if (do_cstm && !do_ptable)
    printf("WARNING: CSTM with no external phrase table\n");

  if (do_lexical)
    Error("the --lexical option is currently not implemented\n");

  if (do_ptable && backward_tm)
    cout << "WARNING: the flag -V has no effect when rescoring with external phrase tables, just use the right score" << endl;


    // read input
  cout <<  "NBest " << cslm_version << " - A tool to process Moses n-best lists" << endl
       << "Copyright (C) 2015 Holger Schwenk, University of Le Mans, France" << endl << endl;
  cout << " - reading input from file '" << in_fname << "'";
  if (in_n>0) cout << " (limited to the first " << in_n << " hypothesis)";
  cout << endl;
  inputfilestream inpf(in_fname);

  if (0 < aux_dim)
    cout << " - reading auxiliary data from file '" << aux_fname << "'" << endl;
  inputfilestream auxf(aux_fname);

    // open source file
  ifstream srcf;
  if (!src_fname.empty()) {
    cout << " - reading source sentences from file " << src_fname << endl;
    srcf.open(src_fname.c_str());
    if (srcf.fail())
      Error("ERROR");
  }


    // open output
  cout << " - writing output to file '" << out_fname << "'";
  if (out_n>0) cout << " (limited to the first " << out_n << " hypothesis)";
  cout << endl;

    // shall we add an offset to the ID ?
  if (offs>0) 
    cout << " - adding offset of " << offs << " to the n-best ids" << endl;
 
  const char *tm_scores_specif_cstr = tm_scores_specif.c_str();

    // shall we rescore with an Moses phrase-table ?
  PtableMosesPtree ptable;
  if (do_ptable & !do_cstm) {
    if (tm_scores_specif.length()<2 || tm_scores_specif[1]!=':')
      Error("format error in the specification of the TM scores");
    tm_nb_scores=tm_scores_specif[0]-'0';
    if (tm_nb_scores<1 || tm_nb_scores>4)
      Error("wrong value for the number of TM scores");
    cout << " - rescoring with an on-disk phrase-table " << pt_fname;
    cout << ", " << tm_nb_scores << " scores";
    if (tm_pos>0) cout << " at position " << tm_pos;
             else cout << " are appended";
    cout << endl;
    ptable.Read(pt_fname, PTABLE_NUM_SCORES,tm_scores_specif_cstr);
    if (!pt_fname2.empty()) ptable.Read(pt_fname2, PTABLE_NUM_SCORES,tm_scores_specif_cstr);
  }
 
 
    // shall we rescore with an CSTM ?
  NbestCSTM cstm;
  if (do_cstm) {
    cout << " - rescoring with CSTM " << cstm_fname;
    if (tm_pos>0) cout << ", scores at position " << tm_pos;
             else cout << ", scores are appended";
    cout << endl;

    if (!pt_fname.empty()) {
      if (tm_scores_specif.length()<2 || tm_scores_specif[1]!=':')
        Error("format error in the specification of the TM scores");
      tm_nb_scores=tm_scores_specif[0]-'0';
      if (tm_nb_scores != 1)
        Error("the external phrase table must provide exactly one score when used as back-off for the CSTM");
      cout << " - using score at position " << (int) (tm_scores_specif[2]-'0') << " from  on-disk phrase-table for out of short-list phrases" << endl;
      if (backward_tm)
        cout << " - calculating backward inverse translation probabilities" << endl;
    }
    cstm.SetSortBehavior(!mach_config.get_unstable_sort());
    cstm.Read((char *)cstm_fname.c_str(), (char *)wl_src_fname.c_str(), (char *)wl_tgt_fname.c_str(),
        pt_fname.empty() ? NULL : (char *)pt_fname.c_str(), PTABLE_NUM_SCORES, (char *)tm_scores_specif_cstr);
  }

 
    // shall we rescore with an LM ?
#ifdef LM_KEN
  NbestLMKEN lm;
#else
#ifdef LM_SRI
  NbestLMSRI lm;
#else
  NbestLM lm;
#endif
#endif
  if (do_lm && !do_cslm) {
    cout << " - rescoring with a " << lm_order << "-gram LM " << lm_fname;
    if (lm_pos>0) cout << ", scores at position " << lm_pos;
             else cout << ", scores are appended";
    cout << endl;
    lm.Read(lm_fname, lm_order);
  }
 
    // shall we rescore with an CSLM ?
  NbestCSLM cslm;
  if (do_cslm) {
    if (wl_fname.empty())
      Error("You need to specify a word-list when rescoring with a CSLM\n");
    if (!do_lm)
      Error("You need to specify a back-off LM when rescoring with a CSLM\n");
    cout << " - rescoring with CSLM " << cslm_fname;
    if (lm_pos>0) cout << ", scores at position " << lm_pos;
             else cout << ", scores are appended";
    cout << endl;
    cslm.SetSortBehavior(!mach_config.get_unstable_sort());
    cslm.Read((char *)cslm_fname.c_str(), (char *)wl_fname.c_str(), (char *)lm_fname.c_str(), tg_pos, aux_dim);
  }


  if (do_calc) cout << " - recalculating global scores" << endl;

    // shall we sort ?
  if (do_sort) cout << " - sorting global scores" << endl;

  const char *w_fname_cstr = w_fname.c_str();
  if (mach_config.get_server()) {
      // server mode 
    if (do_lm || do_ptable || do_cslm || do_cstm || offs!=0)
      Error("no rescoring operations are allowed during server mode");
    if (!do_calc)
      Error("server mode is useless without recalculating the global scores");

    char hostname[MAX_LINE];
    gethostname(hostname, MAX_LINE); hostname[MAX_LINE-1]=0;
    cout << "\nEntering server mode on " << hostname << " pid " << getpid() << endl;

    cout << " - reading the full n-best file into memory" << endl;
    vector<NBest*> nbests;
    int nb_nbest=0;
    while (!inpf.eof()) {
      nbests.push_back(new NBest(inpf, auxf, in_n, do_ptable, 0));
      nb_nbest+=nbests.back()->NbNBest();
      printf (" - %d sents\r", (int) nbests.size());
    }
    inpf.close();
    cout << " - read " << nb_nbest << " n-best hypotheses in " << nbests.size() << " sentences"
       << " (average " << (float) nb_nbest/nbests.size() << ")" << endl;
  
      // wait for new weight vector from named pipe
    cout << " - listing to weight file '" << w_fname << "'" << endl;
    Weights w(w_fname_cstr);
    while (w.ScanLine()>0) {
      cout << " - processing new weights ..."; cout.flush();
      outputfilestream outf(out_fname);
      for (vector<NBest*>::iterator ii=nbests.begin(); ii!=nbests.end(); ii++) {
        if (do_calc) (*ii)->CalcGlobal(w);
        if (do_sort) (*ii)->Sort();
        (*ii)->Write(outf, out_n);
      }
      outf.close();
      cout << " done\n";
    }
    cout << "Leaving server mode" << endl;

    return 0; // TODO free memory
  }
  else {
     // interactive main loop
    outputfilestream outf(out_fname);
     
      // eventually read weights
    Weights w;
    if (!w_fname.empty()) {
      cout << " - reading weights from file '" << w_fname << "'";
      int n=w.Read(w_fname_cstr);
      cout << " (found " << n << " values)" << endl;
      if (lm_pos>0 && lm_pos>n)
        Error("The index for the LM score is out of bounds");
      if (tm_pos>0 && tm_pos+tm_nb_scores>n)
        Error("The index for the TM score is out of bounds");
    }

    time_t t_beg, t_end;
    time(&t_beg);
    int nb_sent=0, nb_nbest=0, nb_phrase=0, nb_phrase_diff=0;
    while (!inpf.eof()) {
      NBest nbest = NBest(inpf, auxf, in_n, do_ptable, aux_dim);
      if (nbest.NbNBest()>0) {
        if (offs!=0) nbest.AddID(offs);
        if (do_calc) nbest.CalcGlobal(w);
        if (do_sort) nbest.Sort();
        if (do_ptable && !do_cstm) nbest.RescorePtable(ptable,srcf,tm_pos);
        if (do_cstm) {
         if (backward_tm)
             nbest.RescorePtableInv(cstm,srcf,tm_pos);
         else
             nbest.RescorePtable(cstm,srcf,tm_pos);
        }
        if (do_lm && !do_cslm) nbest.RescoreLM(lm,lm_pos);
        if (do_cslm) nbest.RescoreLM(cslm,lm_pos);
        nbest.Write(outf, out_n);

        nb_sent++;
        nb_nbest+=nbest.NbNBest();
        nb_phrase+=nbest.NbPhrases();
        nb_phrase_diff+=nbest.NbDiffPhrases();

        printf(" - %d sentences processed\r",nb_sent); fflush(stdout);
      }
    }
    inpf.close();
    outf.close();
    time(&t_end);
    time_t dur=t_end-t_beg;

      // display final statistics
    cout << " - processed " << nb_nbest << " n-best hypotheses in " << nb_sent << " sentences"
         << " (average " << (float) nb_nbest/nb_sent << ")" << endl;
    if (do_ptable || do_cstm) {
      printf(" - with %d phrase pairs (%.2f in average per hypothesis)\n", nb_phrase, (float)nb_phrase/nb_nbest);
      printf(" -      %d were different (%.2f%%), %.2f in average for all n-best of one sentence\n",
        nb_phrase_diff, 100.0*nb_phrase_diff/nb_phrase, (float)nb_phrase_diff/nb_sent);
    }
    if (do_cstm) cstm.Stats();
    if (do_cslm) cslm.Stats();
    cout << " - total time: " << dur/60 << "m" << dur%60 << "s" << endl;
  }

  GpuUnlock();
  return 0;
}
