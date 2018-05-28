                                [SVMTorch II]

   A Support Vector Machine for Large-Scale Regression and Classification
                                  Problems

                               Ronan Collobert
                         (collober@iro.umontreal.ca)
                             (collober@idiap.ch)

                           Universite de Montreal
                       C.P. 6128 Succ. Centre-Ville
                     Montréal, Québec, Canada H3C 3J7

                                     or

         IDIAP, CP 592, rue du Simplon 4, 1920 Martigny, Switzerland

                                 <========>

[o][Description]
SVMTorch II is a new implementation of Vapnik's Support Vector Machine that
works both for classification and regression problems, and that has been
specifically tailored for large-scale problems (such as more than 20000
examples, even for input dimensions higher than 100).

 [New]  What are the differences with the old version ?

   * The code has been completly rewritten in C++. It's now cleaner and
     optimized for a chunk of size 2.
   * Options are a little bit different.
   * The optimality check is now similar to SMO. (Faster)
   * The sparse file format and binary file format are different. Please
     check the manual.
   * A converter is available to convert files from the old format to the
     new one.
   * No more HACK option.
   * The output of the test program in multiclass mode is different.

You can check the ChangeLog file.

For some datasets SVMTorch II is as fast as the old version. However, for
other datasets it is faster or even stupendously faster than the previous
version.
But you now need a C++ compiler; if you do not have access to such compiler,
the old version of SVMTorch is still available here.

[o][Source Code]
As usual, the source code is free for academic use. It must not be modified
or distributed without prior permission of the author. When using SVMTorch
in your scientific work, please cite the following article:

Ronan Collobert and Samy Bengio, Support Vector Machines for Large-Scale
Regression Problems, Journal of Machine Learning Research, vol 1, pp 143-160,
2001.

The software has been successfully compiled on Sun/SOLARIS, Intel/LINUX
operating systems. Your can download it from
ftp.idiap.ch/pub/learning/SVMTorch.tgz.

[o][Try It !!!]

First, you should download  the source code from
ftp.idiap.ch/pub/learning/SVMTorch.tgz and the examples from
ftp.idiap.ch/pub/learning/TrainData.tgz. Note
that the file format for the multiclass example in this archive is not the
same than the previous version. Put this two archive files in the same
directory, and decompress them with

zcat SVMTorch.tgz | tar xf -
zcat TrainData.tgz | tar xf -

It creates two new directories : "SVMTorch" and "TrainData".

Now, go in the "SVMTorch" directory and edit the Makefile. You should only
have to change the following lines, depending on your specific platform :

# C-compiler
CC=g++
#CC=CC
# C-Compiler flags
CFLAGS=-Wall -W -O9 -funroll-all-loops -finline -ffast-math
#CFLAGS=-native -fast -xO5
# linker
LD=g++
#LD=CC
# linker flags
LFLAGS=-Wall -W -O9 -funroll-all-loops -finline -ffast-math
#LFLAGS=-native -fast -xO5
# libraries
LIBS=-lm

The default configuration is set for a machine running with the GNU g++
compiler. An alternate (commented) configuration is proposed for the Sun
Workshop compiler.

Type  "make" and pray.

It should compile without any warning. If you have a problem, please send me
a mail which describes the error and you compiler.

For some platform, you could have to change the include files needed for
"times", a non-standard function used by SVMTorch. You would have to edit
the file "general.h" and change the lines

#ifdef I_WANT_TIME
#include <sys/times.h>
/*#include <limits.h>*/
#include <time.h>
#endif

If it doesn't work or if you don't want to measure the time of the learning
machine, just comment the line :
#define I_WANT_TIME

Note that in "general.h" you can comment the line
#define USEDOUBLE
in order to do the computations in float. IT'S A BAD IDEA : SVMTorch needs
precision.

If everything went well, you should have three programs : "SVMTorch" ,
"SVMTest" and "convert". The first one is the learning machine and the
second one is the testing machine. The last one is to convert files from the
old format into the new one.

If you want to show all the options, just run the programs without any
parameter.

To test the program in classification, try :
SVMTorch ../TrainData/classif_train.dat ../TrainData/model_dummy

It takes less than one minute on a 300Mhz computer. You should have around
913 support vectors (this number could slightly change depending on the
precision of your machine).

To test the SVM on the train data, try :
SVMTest ../TrainData/model_dummy ../TrainData/classif_train.dat
You should have around 0.78% missclassified.

To test the program in regression, try :
SVMTorch -rm -std 900 -eps 20 ../TrainData/regress_train.dat
../TrainData/model_dummy
You should have around 596 support vectors.

Test the model with :
SVMTest ../TrainData/model_dummy ../TrainData/regress_train.dat
The mean squared error should be around 187.27.

To test the program in multiclass and sparse data mode, try :
SVMTorch -multi -sparse -std 7000 ../TrainData/multi_train.dat
../TrainData/model_dummy
It should detect 10 classes (0-9) and the last one should have around 187
support vectors.
(Note that multi_train.dat is 1000 examples taken from the MNIST database)

Test the model with :
SVMTest -sparse -multi ../TrainData/model_dummy ../TrainData/multi_train.dat

You should not have any misclassified example for the multiclass output, and
the model for the class 9 should have 3 missclassified examples.

[o][Options]
The general syntax of SVMTorch and SVMTest is
SVMTorch [options] example_file model_file
SVMTest [options] model_file test_file

Where "example_file" is your training set file, "test_file" is your testing
set file and "model_file" is the SVM-model created by SVMTorch.

All options are described when you launch SVMTorch or SVMTest without any
argument.
By default, SVMTorch is a classification machine. If you want the regression
machine, use option -rm.
Note that the current error displayed by SVMTorch is only an indicator. It
can oscillate.

The default format of SVMTorch and SVMTest is the ASCII format. Therefore,
don't forget to put -bin in the command line if  you  use binary files !!!
No test are made, and you could have an error such as "Check your data" or
"Segmentation fault". Also, don't forget to put -sparse if your data is in
sparse format.

Now you can take a look at the manual.

[o][File Format]
There are two main input formats for "input_file" and "test_file" in
SVMTorch : an ASCII format, and a binary one.

The ASCII format is the following:
<Number n of training/testing samples>  <Dimension d of each sample+1>
<a11> <a12> <a13> .... <a1d> <a1_out>
 .
 .
 .
<an1> <an2> <an3> .... <and> <an_out>

where <aij> is an ASCII floating point number corresponding to the j-th
value of the i-th example and <ai_out> is the i-th desired output (in
classification, it should be +1/-1).

The binary format is the same, without carriage returns.

There is another special input format for SVMTest, when you don't have the
desired output. (To use with the -no option).
The ASCII version of this format is :
<Number n of training/testing samples>  <Dimension d of each sample>
<a11> <a12> <a13> .... <a1d>
 .
 .
 .
<an1> <an2> <an3> .... <and>

The binary format is the same, without carriage returns.

Check the manual for the sparse format, and the multiclass format.

[o][Useful Links]

   * www.kernel-machines.org
   * SVM-Light

  ------------------------------------------------------------------------
Number of visitors:[Counter]
