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

#ifndef _WordList_h
#define _WordList_h

#include <boost/unordered_map.hpp>
#include <vector>
#include "Tools.h"

/**
 * list of words from a vocabulary
 * @note each line of vocabulary files used with this class contains a word and eventually a count and a class number
 */
class WordList
{
public:
  typedef int WordIndex; ///< type for indexes

  /**
   * word information element
   */
  struct WordInfo
  {
    char *word;   ///< UTF8 word
    WordIndex id; ///< sequential number
    int n;        ///< count
    int cl;       ///< class
    WordIndex cl_id; ///< sequential number in the class
  };

  /**
   * iterator through valid word information elements in word list (in sort order if the list is sorted)
   */
  class const_iterator
  {
  public:
    inline const_iterator() : pwlist(NULL) {}
    inline const_iterator(const std::vector<WordInfo*> &wlist, const std::vector<WordInfo*>::const_iterator &iter) : pwlist(&wlist), iter(iter), end(wlist.end()) { if ((iter < end) && ((*iter) == NULL)) (*this)++; }
    inline const_iterator &operator=(const const_iterator &ci) { this->pwlist = ci.pwlist; this->iter = ci.iter; this->end = ci.end; return *this; }
    inline const WordInfo &operator*() const { return **iter; }
    inline const WordInfo *operator->() const { return *iter; }
    inline const_iterator &operator++(int) { for (++iter ; (iter < end) && ((*iter) == NULL) ; iter++); return *this; }
    inline bool operator!=(const const_iterator &ci) const { return ((this->iter != ci.iter) || (this->pwlist != ci.pwlist)); }
  private:
    const std::vector<WordInfo*> *pwlist;
    std::vector<WordInfo*>::const_iterator iter, end;
  };

  static const WordIndex BadIndex ; ///< bad word list index
  static const char *WordUnknown  ; ///< unknown word token
  static const char *WordSentStart; ///< sent start token
  static const char *WordSentEnd  ; ///< sent end token
  static const char *WordPause    ; ///< pause token
  static const char *WordEndOfSeq ; ///< end of sequence token

  /**
   * constructs void word list
   * @param use_tokens use special tokens 'unknown word', 'sent start', 'sent end' and 'pause' (default no)
   */
  WordList(bool use_tokens = false);

  /**
   * destroys word list
   */
  virtual ~WordList();

  /**
   * reads word list from file
   * @note if lines of file contains a count after word, the word list will be sorted by word frequency (most frequent to least frequent)
   * @param fname file name
   * @param use_class sort word list first in order of class and next by frequency
   * @param use_eos use special token 'end of sequence' with maximum frequency (default no)
   * @return number of words read (not the size of word list, depending of special tokens and duplicates)
   */
  inline WordIndex Read(const char *fname, bool use_class, bool use_eos = false) { return WordList::Read(fname, use_class, -1, use_eos); }

  /**
   * reads word list from file
   * @note if lines of file contains a count after word, the word list will be sorted by word frequency (most frequent to least frequent)
   * @param fname file name
   * @param slist_len length of short list (default -1 for all words in file)
   * @param use_eos use special token 'end of sequence' with maximum frequency (default no)
   * @return number of words read (not the size of word list, depending of special tokens and duplicates)
   */
  inline WordIndex Read(const char *fname, WordIndex slist_len = -1, bool use_eos = false) { return WordList::Read(fname, false, slist_len, use_eos); }

  /**
   * checks if word list is sorted by frequency
   * @return true if word list as been sorted during Read method call
   */
  inline bool FrequSort() const {
    return frequ_sort; }

  /**
   * adds a word
   * @note increment word count if already inserted
   * @param word word to insert
   * @return index of word (or WordList::BadIndex in case of overflow)
   */
  WordIndex AddWord(const char *word);

  /**
   * removes a word
   * @param word word to delete
   */
  void RemoveWord(const char *word);

  /**
   * returns an iterator pointing to the first valid element in word list
   */
  inline const_iterator Begin() const {
    return const_iterator(wlist, wlist.begin()); }

  /**
   * returns an iterator referring to the past-the-end element in word list
   */
  inline const_iterator End() const {
    return const_iterator(wlist, wlist.end()); }

  /**
   * writes word list into file
   * @note each line of file will contain a word, a count and eventually a class number
   * @param fname file name
   * @param n_fields number of fields to write: 0/1: word only; 2: word and count; 3: word, count and class
   * @return number of words with count more than zero
   */
  WordIndex Write(const char *fname, unsigned short n_fields) const;

  /**
   * prints statistics on classes
   * @return number of classes
   */
  int ClassStats() const;

  /**
   * gets total number of different words
   * @note removed words are not counted
   */
  inline WordIndex GetSize() const {
    return wordHash.size(); }

  /**
   * checks if 'end of sequence' token is in word list
   * @param word word string
   */
  inline bool HasEOS() const {
    return (wordHash.find(WordList::WordEndOfSeq) != wordHash.cend()); }

  /**
   * gets read index of 'end of sequence' token
   * @return read index or WordList::BadIndex if not found
   */
  inline WordIndex GetEOSIndex() const {
    return GetIndex(WordList::WordEndOfSeq); }

  /**
   * sets number of different words in short list
   */
  inline void SetShortListLength(WordIndex slist_len) {
    if (slist_len >= 0) this->slist_len = slist_len; }

  /**
   * gets number of different words in short list
   */
  inline WordIndex GetShortListLength() const {
    return slist_len; }

  /**
   * gets sum of word counts
   * @param sum_slist out sum of counts in short list
   * @param sum_total out sum of counts in complete list
   */
  void CountWords(ulong &sum_slist, ulong &sum_total) const;

  /**
   * checks if word index is in short list
   * @param index index in word list (sort index if the list is sorted)
   */
  inline bool InShortList(WordIndex index) const {
    return ((index >= 0) && (index < slist_len)); }

  /**
   * checks if word is in short list
   * @param word word string
   */
  inline bool InShortList(const char *word) const {
    WordIndex ri = GetIndex(word);
    return InShortList((frequ_sort && (ri >= 0) && (ri < (WordIndex)sortMap.size())) ? sortMap[ri] : ri); }

  /**
   * gets word information
   * @param read_index read index
   * @return reference to word list element at index if valid or to invalid element (word = NULL, id = WordList::BadIndex, n = 0, cl = 0)
   */
  inline WordInfo& GetWordInfo(WordIndex read_index) {
    WordInfo *wi =(((read_index >= 0) && (read_index < (WordIndex)wlist.size())) ? wlist[frequ_sort ? sortMap[read_index] : read_index] : NULL);
    if (wi != NULL) return *wi;
    else { voidWordInfo.word = NULL; voidWordInfo.id = WordList::BadIndex; voidWordInfo.n = voidWordInfo.cl = 0; return voidWordInfo; } }

  /**
   * gets word information
   * @param index mapped index in word list (sort index if the list is sorted)
   * @return reference to word list element at index if valid or to invalid element (word = NULL, id = WordList::BadIndex, n = 0, cl = 0)
   */
  inline WordInfo& GetWordInfoMapped(WordIndex index) {
    WordInfo *wi =(((index >= 0) && (index < (WordIndex)wlist.size())) ? wlist[index] : NULL);
    if (wi != NULL) return *wi;
    else { voidWordInfo.word = NULL; voidWordInfo.id = WordList::BadIndex; voidWordInfo.n = voidWordInfo.cl = 0; return voidWordInfo; } }

  /**
   * gets frequency sort index corresponding to given read index
   * @param read_index read index
   * @param error_msg error message printed before read index if invalid (default "index")
   * @return sort index, or NULL_WORD if read index is NULL_WORD
   * @note prints error message and exit if read index is invalid
   */
  inline WordIndex MapIndex(WordIndex read_index, const char *error_msg = "index") const {
    if (read_index == NULL_WORD) return NULL_WORD;
    if ((read_index < 0) || (read_index >= (WordIndex)sortMap.size())) ErrorN("%s %d out of range, must be in [0,%zd[", error_msg, read_index, sortMap.size());
    return sortMap[read_index]; }

  /**
   * gets read index corresponding to given word
   * @param word word to find
   * @return word read index or WordList::BadIndex if not found
   */
  inline WordIndex GetIndex(const char *word) const {
    boost::unordered_map<const char*,WordInfo,WordMapHash,WordMapPred>::const_iterator iter = wordHash.find(word);
    if (iter != wordHash.cend()) return iter->second.id; else return WordList::BadIndex; }

  /**
   * gets word corresponding to given index
   * @param index word read index
   * @return word at index if valid or NULL
   */
  inline const char *GetWord(WordIndex index) const {
    WordInfo *wi = (((index >= 0) && (index < (WordIndex)wlist.size())) ? wlist[frequ_sort ? sortMap[index] : index] : NULL);
    return ((wi != NULL) ? wi->word : NULL); }

  /**
   * gets total number of different words in each word class
   * @return vector of sizes (vector length is the number of classes)
   */
  inline const std::vector<int>& GetClassSizes() const {
    return classSizes; }

  /**
   * maps given word class and class index to general frequency sort index
   * @param cl word class
   * @param index word class index
   * @return general sort index or WordList::BadIndex
   */
  inline WordIndex MapIndex(int cl, WordIndex index) const {
    return (((cl >= 0) && (cl < (int)classMap.size()) && (index >= 0) && (index < (WordIndex)classMap[cl].size())) ? classMap[cl][index] : WordList::BadIndex); }

  /**
   * gets total number of different words in each word class
   * @return vector of sizes (vector length is the number of classes)
   */
  inline const void SetSortBehavior(bool val) {
    stable_sort = val; }

private:
  /**
   * reads word list from file
   * @note if lines of file contains a count after word, the word list will be sorted by word frequency (most frequent to least frequent)
   * @param fname file name
   * @param use_class sort word list first in order of class and next by frequency
   * @param slist_len length of short list (-1 for all words in file)
   * @param use_eos use special token 'end of sequence' with maximum frequency
   * @return number of words read (not the size of word list, depending of special tokens and duplicates)
   */
  WordIndex Read(const char *fname, bool use_class, WordIndex slist_len, bool use_eos);

  /**
   * hash function for hash table
   */
  struct WordMapHash : std::unary_function<const char *, std::size_t> {
    std::size_t operator()(const char *) const;
  };

  /**
   * equality function for hash table
   */
  struct WordMapPred : std::binary_function<const char *, const char *, bool> {
    bool operator()(const char *, const char *) const;
  };

  std::vector<const char*> specTokens; ///< vector of special tokens
  std::vector<WordInfo*> wlist; ///< vector of word information elements
  std::vector<WordIndex> sortMap; ///< mapping of indexes from read order to sort order
  std::vector<int> classSizes; ///< table of vocabulary size for each word class
  std::vector<std::vector<WordIndex> > classMap; ///< mapping from class number and class index to general sort index
  boost::unordered_map<const char*,WordInfo,WordMapHash,WordMapPred> wordHash; ///< hash table for mapping words to word information elements
  WordInfo voidWordInfo; ///< void word information element
  WordIndex slist_len; ///< length of short list
  bool use_tokens; ///< use special tokens
  bool frequ_sort; ///< word list sorted by frequency
  bool stable_sort; ///< use a deterministic stable sort w/r to word frequency or class counts, default

  /**
   * inserts special tokens into void word list
   * @return next available index in word list
   */
  WordIndex InsertTokens();
};

#endif
