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
#include <stdexcept>
#include <stdlib.h>
#include "Tools.h"
#include "Toolsgz.h"


int Weights::Read(const char *fname)
{
  ifstream wftmp;
  wftmp.open(fname);
  if (wftmp.fail())
    Error("ERROR");

  float f;
  while (wftmp >> f) val.push_back(f);  

  wftmp.close();
  return val.size();
}

Weights::Weights(const char *fname)
{
  wf.open(fname);
  if (wf.fail())
    Error("ERROR");
}

int Weights::ScanLine()
{
  if (wf.eof()) Error("Weights::ScanLine() called without open file");

  string line;
  getline(wf,line);
  if (!wf.good()) return 0;
  printf("scan: got '%s'\n",line.c_str());
  if (line.size()==0) return 0;
  
    // parse the values
  uint pos=0, epos;
  val.clear();
  while ((epos=line.find(WEIGHT_DELIM,pos))<100000) {
    val.push_back(Scan<float>(line.substr(pos,epos-pos)));
    pos=epos+strlen(WEIGHT_DELIM);
  }
  val.push_back(Scan<float>(line.substr(pos,line.size())));
  printf("scan: parsed into %d values\n",(int) val.size());

  wf.clear();
  return val.size();
}

//
//
//

void gzifstream::open(char *fname)
{
  //check if file is readable
  std::filebuf* fb = new std::filebuf();
  _fail=(fb->open(fname, std::ios::in)==NULL);

  char *sptr=strrchr(fname,'.');
  if (sptr && strcmp(sptr,".gz")==0) {
    fb->close(); delete fb;
    gz_streambuf = new gzfilebuf(fname);
  } else {
    gz_streambuf = fb;
  }
  this->init(gz_streambuf);
}


void gzofstream::open(char *fname)
{
  //check if file is readable
  std::filebuf* fb = new std::filebuf();
  _fail=(fb->open(fname, std::ios::out)==NULL);
  //cerr << "fail: " << _fail <<endl;

  char *sptr=strrchr(fname,'.');
  if (sptr && strcmp(sptr,".gz")==0) {
    fb->close(); delete fb;
    gz_streambuf = new gzfilebuf(fname);
  } else {
    gz_streambuf = fb;
  }
  this->init(gz_streambuf);
}

//
//
//

inputfilestream::inputfilestream(const std::string &filePath)
: std::istream(0),
m_streambuf(0)
{
  //check if file is readable
  std::filebuf* fb = new std::filebuf();
  _good=(fb->open(filePath.c_str(), std::ios::in)!=NULL);
  
  if (filePath.size() > 3 &&
      filePath.substr(filePath.size() - 3, 3) == ".gz")
  {
    fb->close(); delete fb;
    m_streambuf = new gzfilebuf(filePath.c_str());  
  } else {
    m_streambuf = fb;
  }
  this->init(m_streambuf);
}

inputfilestream::~inputfilestream()
{
  delete m_streambuf; m_streambuf = 0;
}

void inputfilestream::close()
{
}

outputfilestream::outputfilestream(const std::string &filePath)
: std::ostream(0),
m_streambuf(0)
{
  //check if file is readable
  std::filebuf* fb = new std::filebuf();
  _good=(fb->open(filePath.c_str(), std::ios::out)!=NULL);  
  
  if (filePath.size() > 3 && filePath.substr(filePath.size() - 3, 3) == ".gz")
  {
    fb->close(); delete fb;
    m_streambuf = new gzfilebuf(filePath.c_str(), 0);
  } else {
    m_streambuf = fb;
  }
  this->init(m_streambuf);
}

outputfilestream::~outputfilestream()
{
  delete m_streambuf; m_streambuf = 0;
}

void outputfilestream::close()
{
}

