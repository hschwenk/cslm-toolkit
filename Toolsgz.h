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

#ifndef TOOLSGZ_H
#define TOOLSGZ_H

using namespace std;

#include <stdexcept>
#include <limits>
#include <cstring>	// for memmove
#include <zlib.h>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <streambuf>

#define US_NOSET (numeric_limits<unsigned short>::max())
#define MAX_LINE  1024


//
//
//

#define WEIGHT_DELIM " "

class Weights {
  vector<float> val;
  ifstream wf;
 public:
  Weights() : wf(NULL) {};
  Weights(const char *fname);
  ~Weights() {if (wf) wf.close(); };
  int Read(const char *);
  int ScanLine();
  friend class Hypo;
};


class gzfilebuf : public std::streambuf {
public:
  gzfilebuf(const char *filename)
  { _gzf = gzopen(filename, "rb"); 
    setg (_buff+sizeof(int),     // beginning of putback area
          _buff+sizeof(int),     // read position
          _buff+sizeof(int));    // end position
  }
  gzfilebuf(const char *filename, int dummy)
  { _gzf = gzopen(filename, "wb");
    setp (_buff,                 // write position
          _buff+_buffsize);      // end position
  }
  ~gzfilebuf()
  {
    sync();
    gzclose(_gzf);
  }

protected:
  // synchronize stream buffer
  virtual int sync() {
    if (pptr() > pbase()) {
      // write characters in buffer
      int num = gzwrite(_gzf, _buff, pptr() - pbase());
      if (num < 0)
          return -1;

      // reset _buff pointers
      setp (_buff,               // write position
            _buff+_buffsize);    // end of buffer
    }
    return 0;
  }

  // write one character
  virtual int_type overflow (int_type c) {
    // is write position before end of _buff?
        if (pptr() < epptr()) {
          (*pptr()) = traits_type::to_char_type(c);
          return c;
        }

        // write new characters
        int num = gzwrite(_gzf, _buff, _buffsize);
        if (num <= 0) {
            // ERROR or EOF
            return EOF;
        }

        // reset _buff pointers
        setp (_buff,               // write position
              _buff+_buffsize);    // end of buffer

        // write character
        (*pptr()) = traits_type::to_char_type(c);
        return c;
  }

  // write multiple characters
  virtual
  std::streamsize xsputn (const char* s,
                          std::streamsize num) {
    sync();
    return gzwrite(_gzf,s,num);
  }

  // set internal position pointer to absolute position
  virtual std::streampos seekpos ( std::streampos sp, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out ){ throw;
  }

  // read one character
  virtual int_type underflow () {
    // is read position before end of _buff?
        if (gptr() < egptr()) {
            return traits_type::to_int_type(*gptr());
        }

        /* process size of putback area
         * - use number of characters read
         * - but at most four
         */
        unsigned int numPutback = gptr() - eback();
        if (numPutback > sizeof(int)) {
            numPutback = sizeof(int);
        }

        /* copy up to four characters previously read into
         * the putback _buff (area of first four characters)
         */
        memmove (_buff+(sizeof(int)-numPutback), gptr()-numPutback,
                      numPutback);

        // read new characters
        int num = gzread(_gzf, _buff+sizeof(int), _buffsize-sizeof(int));
        if (num <= 0) {
            // ERROR or EOF
            return EOF;
        }

        // reset _buff pointers
        setg (_buff+(sizeof(int)-numPutback),   // beginning of putback area
              _buff+sizeof(int),                // read position
              _buff+sizeof(int)+num);           // end of buffer

        // return next character
        return traits_type::to_int_type(*gptr());
  }

  // read multiple characters
  std::streamsize xsgetn (char* s,
                          std::streamsize num) {
    return gzread(_gzf,s,num);
  }

private:
  gzFile _gzf;
  static const unsigned int _buffsize = 1024;
  char _buff[_buffsize];
};

//
//
//

class inputfilestream : public std::istream
{
protected:
        std::streambuf *m_streambuf;
	bool _good;
public:
  
        inputfilestream(const std::string &filePath);
        ~inputfilestream();
	bool good(){return _good;}
        void close();
};

class outputfilestream : public std::ostream
{
protected:
        std::streambuf *m_streambuf;
	bool _good;
public:
  
        outputfilestream(const std::string &filePath);
        ~outputfilestream();
	bool good(){return _good;}
        void close();
};

/****************************************************** 
 *
 * Compressed File IO
 */

class gzifstream : public std::istream
{
protected:
  std::streambuf *gz_streambuf;
  bool _fail;
public:
  gzifstream() : gz_streambuf(0), _fail(true) {};
  ~gzifstream() {if (gz_streambuf) delete(gz_streambuf); };
  void open(char*);
  bool fail() {return _fail;}
  void close() {};
};


class gzofstream : public std::ostream
{
protected:
  std::streambuf *gz_streambuf;
  bool _fail;
public:
  gzofstream() : gz_streambuf(0), _fail(true) {};
  ~gzofstream() {if (gz_streambuf) delete(gz_streambuf); };
  void open(char*);
  bool fail() {return _fail;}
  void close() {};
};


//****************************************************** 


template<typename T>
inline T Scan(const std::string &input)
{
         std::stringstream stream(input);
         T ret;
         stream >> ret;
         return ret;
}


#endif

