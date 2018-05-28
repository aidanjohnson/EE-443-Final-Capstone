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


#include "general.h"
#include "IOTorch.h"
#include "OldIOTorch.h"
#include "string.h"

typedef struct
{
    bool sparse_mode;
    bool input_bin;
    bool output_bin;
    bool target;
    
} parametres;

void help(char **argv)
{
  cout << "\n* SVMTorch converter" << endl;
  cout << "* a part of " << VERSION << endl;
  cout << "* \n* usage: " << argv[0] << " [options] <old format file> <new format file>" << endl;
  cout << "* \n* \n";
  cout << "* Options:" << endl;
  cout << "*   -help   this little help" << endl;
  cout << "*   -ibin   input is in binary format (default is false)" << endl;
  cout << "*   -obin   output is in binary format (default is false)" << endl;
  cout << "*   -sparse sparse mode (default is false)" << endl;
  cout << "*   -no     no target inside (default is false)" << endl;
  cout << "* \n* \n* Make your choice.\n\n";
  exit(0);
}

void scan_cmd(int argc, char **argv, parametres *params)
{  
  int i = 1;
  int erreur;
  int maxa;
  
  if(argc < 3)
    help(argv);
  
  maxa = argc-2;
  while(i < maxa)
  {
    erreur = 1;
    
    if(!strcmp(argv[i], "-help"))
      help(argv);

    if(!strcmp(argv[i], "-ibin"))
    {
      params->input_bin = true;
      erreur = 0;
    }

    if(!strcmp(argv[i], "-obin"))
    {
      params->output_bin = true;
      erreur = 0;
    }    

    if(!strcmp(argv[i], "-no"))
    {
      params->target = false;
      erreur = 0;
    }    

    if(!strcmp(argv[i], "-sparse"))
    {
      params->sparse_mode = true;
      erreur = 0;
    }

    i++;    
    if(erreur)
      help(argv);
  }
}

int main(int argc, char **argv)
{
  parametres params;

  params.sparse_mode = false;
  params.input_bin = false;
  params.output_bin = false;
  params.target = true;

  scan_cmd(argc, argv, &params);

  IOTorch nf;
  OldIOTorch of;
  real *y;
  int l, w;

  if(params.sparse_mode)
  {
    sreal **data;
    of.loadData(argv[argc-2], &data, &y, l, w, params.input_bin, params.target);
    cout << "# We have read " << l << " lines with dimension " << w;
    if(params.target)
      cout << " with the target" << endl;
    else
      cout << endl;
    cout << "# Now I write that..." << endl;
    nf.saveData(argv[argc-1],  data,  y, l, w, params.output_bin, -1, params.target);
  }
  else
  {
    real **data;
    of.loadData(argv[argc-2], &data, &y, l, w, params.input_bin, params.target);
    cout << "# We have read " << l << " lines with dimension " << w;
    if(params.target)
      cout << " with the target" << endl;
    else
      cout << endl;
    cout << "# Now I write that..." << endl;
    nf.saveData(argv[argc-1],  data,  y, l, w, params.output_bin, -1, params.target);
  }

  cout << "\n# Yeah.\n\n";
  return(0);
}
