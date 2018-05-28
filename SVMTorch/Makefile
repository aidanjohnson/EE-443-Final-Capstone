#/===////////////////////////////////////////////////////////////////////
#/                                                                     //
#/   SVMTorch II V1.77 [Ennastie]                                      //
#/                                                                     //
#/   Author: Ronan Collobert                                           //
#/   Date: 20.11.2001                                                  //
#/                                                                     //
#/   Copyright (c) 2001  IDIAP & DIRO U. de Montreal                   //
#/                       All rights reserved                           //
#/                                                                     //
#/   This software is available for non-commercial use only. It must   //
#/   not be modified and distributed without prior permission of the   //
#/   author. The author is not responsible for implications from the   //
#/   use of this software.                                             //
#/                                                                     //
#///////////////////////////////////////////////////////////////////===//


# C-compiler
CC=g++
#CC=CC
# C-Compiler flags
CFLAGS=-Wall -W -O9 -funroll-all-loops -finline -ffast-math -fopenmp
#CFLAGS=-native -fast -xO5
# linker
LD=g++
#LD=CC
# linker flags
LFLAGS=-Wall -W -O9 -funroll-all-loops -finline -ffast-math -fopenmp
#LFLAGS=-native -fast -xO5
# libraries
LIBS=-lm


all:	bordel SVMTorch SVMTest convert okb

clean:
	rm -f *.o 
	rm -f SVMTorch
	rm -f SVMTest
	rm -f convert

help:   
	@echo
	@echo "# SVMTorch II"
	@echo
	@echo "# (c) Ronan Collobert 2001"
	@echo 
	@echo "# usage: make all | clean | SVMTorch | SVMTest | convert"
	@echo 
	@echo "   # all                 builds all [Default]"
	@echo "   # clean               removes object files"
	@echo "   # SVMTorch            builds learning program"
	@echo "   # SVMTest             builds testing program"
	@echo "   # convert             builds converter"
	@echo

bordel:
	@echo
	@echo "# Try to compile SVMTorch"
	@echo

okb:
	@echo
	@echo "# OK."
	@echo


Cache.o: Cache.h Cache.cc
	$(CC) -c $(CFLAGS) Cache.cc -o Cache.o

IOTorch.o: IOTorch.h IOTorch.cc
	$(CC) -c $(CFLAGS) IOTorch.cc -o IOTorch.o

Kernel.o: Kernel.h Kernel.cc
	$(CC) -c $(CFLAGS) Kernel.cc -o Kernel.o

OldIOTorch.o: OldIOTorch.h OldIOTorch.cc
	$(CC) -c $(CFLAGS) OldIOTorch.cc -o OldIOTorch.o

SVM.o: SVM.h SVM.cc
	$(CC) -c $(CFLAGS) SVM.cc -o SVM.o

SVMTest.o: SVMTest.cc
	$(CC) -c $(CFLAGS) SVMTest.cc -o SVMTest.o

SVMTorch.o: SVMTorch.cc
	$(CC) -c $(CFLAGS) SVMTorch.cc -o SVMTorch.o

StandardSVM.o: StandardSVM.h StandardSVM.cc
	$(CC) -c $(CFLAGS) StandardSVM.cc -o StandardSVM.o

UserKernel.o: UserKernel.h UserKernel.cc
	$(CC) -c $(CFLAGS) UserKernel.cc -o UserKernel.o

convert.o: convert.cc
	$(CC) -c $(CFLAGS) convert.cc -o convert.o

general.o: general.h general.cc
	$(CC) -c $(CFLAGS) general.cc -o general.o

SVMTorch: SVMTorch.o SVM.o StandardSVM.o Cache.o Kernel.o IOTorch.o general.o UserKernel.o
	$(LD) $(LFLAGS) -o SVMTorch SVMTorch.o SVM.o StandardSVM.o Cache.o Kernel.o IOTorch.o general.o UserKernel.o

SVMTest: SVMTest.o SVM.o StandardSVM.o Cache.o Kernel.o IOTorch.o general.o UserKernel.o
	$(LD) $(LFLAGS) -o SVMTest SVMTest.o SVM.o StandardSVM.o Cache.o Kernel.o IOTorch.o general.o UserKernel.o

convert: general.o IOTorch.o OldIOTorch.o convert.o
	$(LD) $(LFLAGS) -o convert general.o IOTorch.o OldIOTorch.o convert.o
