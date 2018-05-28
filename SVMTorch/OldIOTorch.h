//===////////////////////////////////////////////////////////////////////
//                                                                     //
//   SVMTorch II V1.77 [Ennastie]                                      //
//                                                                     //
//   Author: Ronan Collobert                                           //
//   Date: 20.11.2001                                                  //
//                                                                     //
//   Copyright (c) 2001  IDIAP & DIRO U. de Montreal                   //
//                       All rights reserved                           //
//                                                                     //
//   This software is available for non-commercial use only. It must   //
//   not be modified and distributed without prior permission of the   //
//   author. The author is not responsible for implications from the   //
//   use of this software.                                             //
//                                                                     //
////////////////////////////////////////////////////////////////////===//


#ifndef OLDIOTORCH_INC
#define OLDIOTORCH_INC

#include "general.h"

class OldIOTorch
{
  private:
  void error();

  public:

  void loadData(string file,  real ***data_, real **y_, int &l, int &w, bool bin, bool target=true);
  void loadData(string file, sreal ***data_, real **y_, int &l, int &w, bool bin, bool target=true);
};

#endif
