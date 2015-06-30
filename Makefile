#
# Example of a simple makefile to build the CSLM library and associated tools
#
# rcsid $Id: Makefile,v 1.81 2015/01/12 11:34:04 aransa Exp $
#
# CONFIGURATION
#  1) select platform and math libraries by decommenting the corresponding lines below
#      -DBLAS_ATLAS	free implementation of BLAS routines
#			needs compilation of Blas.c
#      -DBLAS_INTEL_MKL	more efficient for INTEL processors
#			neither Blas.c nor Gpu.cu should be used
#      CUDA=1		support for Nvidia GPU cards
#			needs compilation of Gpu.cu
#  2) choose back-off LM tool
#      BOLM_TOOL=KENLM (default)
#      BOLM_TOOL=SRILM
#  3) choose to use or not CSTM with Moses (default not)
#  4) choose to build or not LM tools (default yes)
#  5) adapt the optimization to your CPU architecture
#      -mtune=core2	good value for current Intel Nethalem CPUs
#      -mtune=native	if you use the executable on a specific processor
#			The most optimized version, but not portable.


# git stuff
LAST_TAG_COMMIT = $(shell git rev-list --tags --max-count=1)
LAST_TAG	= $(shell git describe --tags $(LAST_TAG_COMMIT) )
# Versioning
VERSIONFILE   = VERSION
VERSION       = $(shell [ -f $(VERSIONFILE) ] && head $(VERSIONFILE) || echo "0.0.1")
# OR try to guess directly from the last git tag
#VERSION    = $(shell  git describe --tags $(LAST_TAG_COMMIT) | sed "s/^$(TAG_PREFIX)//")
MAJOR	      = $(shell echo $(VERSION) | sed "s/^\([0-9]*\).*/\1/")
MINOR	      = $(shell echo $(VERSION) | sed "s/[0-9]*\.\([0-9]*\).*/\1/")
PATCH	      = $(shell echo $(VERSION) | sed "s/[0-9]*\.[0-9]*\.\([0-9]*\).*/\1/")
# total number of commits
BUILD	    = $(shell git log --oneline | wc -l | sed -e "s/[ \t]*//g")
#REVISION   = $(shell git rev-list $(LAST_TAG).. --count)
NEXT_MAJOR_VERSION = $(shell expr $(MAJOR) + 1).0.0-b$(BUILD)
NEXT_MINOR_VERSION = $(MAJOR).$(shell expr $(MINOR) + 1).0-b$(BUILD)
NEXT_PATCH_VERSION = $(MAJOR).$(MINOR).$(shell expr $(PATCH) + 1)-b$(BUILD)



.PHONY: all versioninfo
.DEFAULT: all
CUDA_ROOT ?= /opt/cuda
CUDA ?= 0
# K20: sm_35 / M2090: sm_20 / GTX690: sm_30 / GTX580: sm_20
NVCC_FLAGS ?= -g -arch=sm_35 -use_fast_math

SRCS=Tools.cpp Toolsgz.cpp \
	MachConfig.cpp \
	Mach.cpp MachTab.cpp \
	MachCopy.cpp MachLin.cpp MachSig.cpp MachTanh.cpp MachSoftmax.cpp MachSoftmaxStable.cpp MachLinRectif.cpp \
	MachMulti.cpp MachSeq.cpp MachPar.cpp MachSplit.cpp MachSplit1.cpp MachJoin.cpp \
	Data.cpp DataFile.cpp DataAscii.cpp DataAsciiClass.cpp DataMnist.cpp DataNgramBin.cpp DataPhraseBin.cpp \
	ErrFct.cpp ErrFctMSE.cpp ErrFctMCE.cpp ErrFctCrossEnt.cpp ErrFctSoftmCrossEntNgram.cpp ErrFctSoftmCrossEntNgramMulti.cpp \
	Hypo.cpp Lrate.cpp NbestLM.cpp NbestCSLM.cpp \
	Trainer.cpp TrainerNgram.cpp TrainerNgramSlist.cpp \
	MachSoftmaxClass.cpp ErrFctSoftmClassCrossEntNgram.cpp TrainerNgramClass.cpp \
	Shareable.cpp \
	WordList.cpp

SRCS+=MachCombined.cpp MachAvr.cpp	# experimental

TOOLS=cslm_train cslm_eval nn_train nn_info text2bin extract2bin cslm_rescore mach_dump cslm_ngrams dumpEmbeddings
TOOLS+=conv_wl_sort

ifeq "$(CUDA)" "0"
	SRCS+=Blas.c
else
	SRCS+=Gpu.cu
endif
#
# select which BLAS library to use
#
ifeq ($(CUDA), 0)
  # default Atlas BLAS available on many LINUX distrubutions
  # This is slower than Intel's MKL
  #MKL_ROOT=/usr/lib64/atlas
  #BLAS=-DBLAS_ATLAS
  #LIBS_MKL=-L${MKL_ROOT} -lptf77blas
  #LIBS_MKL=${MKL_ROOT}/libptf77blas.so.3

  # Intel's MKL libray (http://software.intel.com/en-us/intel-mkl)
  # This is usually much faster and offers better support of multi-threading
  #
  MKL_ROOT=/opt/intel/composer_xe_2013_sp1.2.144
  BLAS=-DBLAS_INTEL_MKL -I${MKL_ROOT}/mkl/include
  LIBS_MKL=-L$(MKL_ROOT)/mkl/lib/intel64 -L$(MKL_ROOT)/mkl/lib/em64t -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -L$(MKL_ROOT)/lib/intel64 -liomp5 -lpthread
else
  # CUDA BLAS libray to run on Nvidia GPU cards
  # Needs dedicated hardware and installation of Nvidia's drivers and CUDA libaries
  BLAS=-DBLAS_CUDA -I${CUDA_ROOT}/include
  LIBS_MKL=-L${CUDA_ROOT}/lib64 -lcublas -lnpps -lnppc -lcudart -lcurand -lnvidia-ml -lpthread
endif

# choose implementation of back-off LM
#
BOLM_TOOL=KENLM
#BOLM_TOOL=SRILM
ifeq ($(BOLM_TOOL),KENLM)
	# KENLM
	# set KenLM directory which must contain subdirectories lm, util and util/double-conversion
	# (remove files lm/*_test.cc and util/*_test.cc which are unused)
	BOLM_DIR=kenlm
	# set max ngram order to KENLM_MAX_ORDER
	BOLM_FLAGS=-I$(BOLM_DIR) -DLM_KEN -DKENLM_MAX_ORDER=9
	BOLM_ARCH=$(BOLM_DIR)/libkenlm.a
	BOLM_SRCS=$(wildcard $(BOLM_DIR)/lm/*.cc)
	BOLM_SRCS+=$(wildcard $(BOLM_DIR)/util/*.cc)
	BOLM_SRCS+=$(wildcard $(BOLM_DIR)/util/double-conversion/*.cc)
	BOLM_OBJS:=$(BOLM_SRCS:.cc=.o)
	SRCS+=NbestLMKEN.cpp BackoffLmKen.cpp
endif
ifeq ($(BOLM_TOOL),SRILM)
	# SRILM
	# the environment variable SRILM must be correctly set
	BOLM_DIR=$(SRILM)
	BOLM_FLAGS=-I$(BOLM_DIR)/include -DLM_SRI
	LM_LIBS=-L$(BOLM_DIR)/lib/i686 -loolm -ldstruct -lmisc
	SRCS+=NbestLMSRI.cpp BackoffLmSri.cpp
endif

# Moses 
# needed to access binary phrase-tables for CSTM
CSTM = 0
ifeq ($(CSTM), 1)
	MOSES_ROOT=/opt/mt/moses-smt
	MOSES_INC=-I$(MOSES_ROOT)/mosesdecoder-2013-11-15 -I$(MOSES_ROOT)/mosesdecoder-2013-11-15/moses
	MOSES_CFLAGS=-DCSTM -DMAX_NUM_FACTORS=4
	MOSES_LIB=-L$(MOSES_ROOT)/moses-2013-11-15/lib -lmoses
	SRCS+=NBest.cpp NbestCSTM.cpp TrainerPhraseSlist.cpp TrainerPhraseSlist1.cpp PtableMosesPtree.cpp
	TOOLS+=cstm_train cstm_eval nbest
else
	MOSES_INC=
	MOSES_CFLAGS=
endif

# LM tools
LM_TOOLS = 0
ifeq ($(LM_TOOLS), 1)
	LM_TOOLS_DIR=lm_tools
	LM_TOOLS_INC=-I$(LM_TOOLS_DIR)/include/lmmax -I$(LM_TOOLS_DIR)/include/sphinx3 -I$(LM_TOOLS_DIR)/include/sphinxbase
	LM_TOOLS_SRCS_main=$(wildcard $(LM_TOOLS_DIR)/libsphinxbase/util/*.c)
	LM_TOOLS_SRCS_main+=$(wildcard $(LM_TOOLS_DIR)/libs3decoder/libam/*.c)
	LM_TOOLS_SRCS_main+=$(wildcard $(LM_TOOLS_DIR)/libs3decoder/libcommon/*.c)
	LM_TOOLS_SRCS_main+=$(wildcard $(LM_TOOLS_DIR)/libs3decoder/libdict/*.c)
	LM_TOOLS_SRCS_main+=$(wildcard $(LM_TOOLS_DIR)/libs3decoder/liblm_Ng/*.c)
	LM_TOOLS_SRCS_lmmax=$(wildcard $(LM_TOOLS_DIR)/liblmmax/*.c)
	LM_TOOLS_SRCS_search=$(wildcard $(LM_TOOLS_DIR)/libs3decoder/libsearch/*.c*)
	LM_TOOLS_OBJS_main:=$(LM_TOOLS_SRCS_main:.c=.o)
	LM_TOOLS_OBJS_lmmax:=$(LM_TOOLS_SRCS_lmmax:.c=.o) $(LM_TOOLS_OBJS_main)
	LM_TOOLS_OBJS_search:=$(LM_TOOLS_SRCS_search:.c=.o) $(LM_TOOLS_OBJS_lmmax)
	LM_TOOLS_OBJS_search:=$(LM_TOOLS_OBJS_search:.cpp=.o)
	LM_TOOLS_OBJS:=$(wildcard $(LM_TOOLS_DIR)/*.o) $(LM_TOOLS_OBJS_search)
	TOOLS+=$(LM_TOOLS_DIR)/lm_convert $(LM_TOOLS_DIR)/dmp2max $(LM_TOOLS_DIR)/max_read $(LM_TOOLS_DIR)/s3astarLRU
endif

LM_LIBS+=-lz -lm

LIB_CSLM=libcslm.a
LIBS=$(LIBS_MKL) $(LM_LIBS) -lboost_program_options

NVCC=${CUDA_ROOT}/bin/nvcc ${DB} ${NVCC_FLAGS}

CC=g++
OPT_FLAGS?=-mtune=native -march=native -O3 -Ofast
#  core2:	no sse4_1, sse4_2
#  corei7:	eg. Intel X5675, Core i7 with sse4_2, aes, pclmulqdq)
#  corei7-avx:	eg. Intel E5-2670 which adds avx
#  corei7-avx-i:	eg. Intel E5-2690v2 which adds avx
CFLAGS=${OPT_FLAGS} -Wall -g ${DB} ${BLAS} ${BOLM_FLAGS} ${MOSES_INC} ${MOSES_CFLAGS}

OBJS:=$(SRCS:.cpp=.o)
OBJS:=$(OBJS:.cu=.o)
OBJS:=$(OBJS:.c=.o)

all: $(TOOLS)


ifeq ($(LM_TOOLS), 1)

$(LM_TOOLS_DIR)/%.o: $(LM_TOOLS_DIR)/%.c
	gcc -g -O3 -w $(LM_TOOLS_INC) -o $@ -c $<

$(LM_TOOLS_DIR)/%.o: $(LM_TOOLS_DIR)/%.cpp
	$(CC) ${OPT_FLAGS} -g ${BLAS} ${BOLM_FLAGS} -I. $(LM_TOOLS_INC) -o $@ -c $<

$(LM_TOOLS_DIR)/dmp2max: $(LM_TOOLS_DIR)/dmp2max.o $(LM_TOOLS_OBJS_lmmax)
	gcc -g -O3 -Wall -lm -pthread -o $@ $(LM_TOOLS_DIR)/dmp2max.o $(LM_TOOLS_OBJS_lmmax)

$(LM_TOOLS_DIR)/lm_convert: $(LM_TOOLS_DIR)/main_lm_convert.o $(LM_TOOLS_OBJS_main)
	gcc -g -O3 -Wall -lm -pthread -o $@ $(LM_TOOLS_DIR)/main_lm_convert.o $(LM_TOOLS_OBJS_main)

$(LM_TOOLS_DIR)/max_read: $(LM_TOOLS_DIR)/lm_max_read.o $(LM_TOOLS_OBJS_lmmax)
	gcc -g -O3 -Wall -lm -pthread -o $@ $(LM_TOOLS_DIR)/lm_max_read.o $(LM_TOOLS_OBJS_lmmax)

$(LM_TOOLS_DIR)/s3astarLRU: $(LM_TOOLS_DIR)/main_astar.o $(LM_TOOLS_OBJS_search) $(LIB_CSLM) $(BOLM_TOOL)
	$(CC) ${OPT_FLAGS} -g -Wall -o $@ $(LM_TOOLS_DIR)/main_astar.o $(LM_TOOLS_OBJS_search) $(LIB_CSLM) $(BOLM_ARCH) $(LM_LIBS) $(LIBS_MKL)

endif

########################### Version Management #############################
versioninfo:	
	@echo "Version file: $(VERSIONFILE)"
	@echo "Current version: $(VERSION)"
#@echo "(major: $(MAJOR), minor: $(MINOR), patch: $(PATCH))"
	@echo "Last tag: $(LAST_TAG)"
	@echo "$(shell git rev-list $(LAST_TAG).. --count) commit(s) since last tag"
	@echo "Build: $(BUILD) (total number of commits)"
	@echo "next major version: $(NEXT_MAJOR_VERSION)"
	@echo "next minor version: $(NEXT_MINOR_VERSION)"
	@echo "next patch version: $(NEXT_PATCH_VERSION)"



%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

PtableMosesPtree.o: PtableMosesPtree.cpp PtableMosesPtree.h

TrainerPhraseSlist.o: TrainerPhraseSlist.cpp TrainerPhraseSlist.h

%.o: %.cu
	${NVCC} -I${CUDA_ROOT}/include -c $<

$(BOLM_DIR)/%.o: $(BOLM_DIR)/%.cc
	$(CC) ${OPT_FLAGS} -w ${BOLM_FLAGS} -DNDEBUG -DHAVE_ZLIB -o $@ -c $<

%: %.cpp $(LIB_CSLM)
	$(CC) $(CFLAGS) -o $@ $< $(LIB_CSLM) $(LIBS)

Blas.o: Blas.c
	gcc -O3 -mtune=core2 -mfpmath=sse -msse4.2 -c Blas.c

$(BOLM_TOOL): $(BOLM_OBJS)
ifneq ($(BOLM_ARCH),)
	ar r $(BOLM_ARCH) $(BOLM_OBJS)
	touch ${BOLM_TOOL}
endif

$(LIB_CSLM): $(OBJS)
	ar r $(LIB_CSLM) $(OBJS)



cslm_train cslm_eval cslm_rescore: %: %.cpp $(LIB_CSLM) $(BOLM_TOOL)
	${CC} $(CFLAGS) -o $@ $< $(LIB_CSLM) $(BOLM_ARCH) $(LIBS)

cstm_deep_train cstm_train cstm_train1 cstm_eval: %: %.cpp $(LIB_CSLM)
	${CC} $(CFLAGS) -o $@ $< $(LIB_CSLM) $(LIBS) $(MOSES_LIB) -lrt

nbest: nbest_cmd.cpp $(LIB_CSLM) $(BOLM_TOOL)
	${CC} $(CFLAGS) -o nbest nbest_cmd.cpp $(LIB_CSLM) $(BOLM_ARCH) $(LIBS) $(MOSES_LIB) -lrt

ptable queryPhraseTable: %: %.cpp
	${CC} $(MOSES_INC) -o $@ $< $(MOSES_LIB) -lrt

clean:
	@rm -f $(OBJS) $(BOLM_OBJS) $(LM_TOOLS_OBJS) $(TOOLS) $(LIB_CSLM) $(BOLM_ARCH)

#Must be run without CUDA=1
depend:
	echo "" > .depend
	for i in $(SRCS); do\
	  ${CC} -MM -MG $$i >> .depend;\
	done

locks: RCS
	@echo "current RCS locks:\n"
	@grep 'strict;' RCS/* | grep -v locks
	@echo ""

diff: RCS
	@echo "changed files with respect to last RCS branch:\n"
	@for i in RCS/*,v RCS/.depend,v; \
	  do rcsdiff -q $$i > /dev/null; \
	    if [ $$? = 1 ]; then echo $$i; fi; \
	done
	@echo ""


include .depend
Gpu.o: Gpu.cu Gpu.cuh Tools.h
# DO NOT DELETE
