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


#include "OldIOTorch.h"

void OldIOTorch::error()
{
  cout << "$ File error. Check your data.\n\n";
  exit(0);
}

void OldIOTorch::loadData(string file, real ***data_, real **y_,
                       int &l, int &w, bool bin, bool target)
{  
  real **data = NULL;
  real *y = NULL;

  ifstream f(file.c_str());

  if(!f)
    error();

  if(bin)
  {
    f.read((char *)&l, sizeof(int));
    f.read((char *)&w, sizeof(int));
  }
  else
  {
    f >> l;
    f >> w;
    if(target)
      w -= 1;
  }
  
  data = new real*[l];
  for(int i = 0; i < l; i++)
    data[i] = new real[w];
  
  if(target)
    y = new real[l];
  
  if(bin)
  {    
    for(int i = 0; i < l; i++)
      f.read((char *)data[i], sizeof(real)*w);

    if(target)
      f.read((char *)y, sizeof(real)*l);
    
  }
  else
  {
    for(int i = 0; i < l; i++)
    {
      for(int j = 0; j < w; j++)
        f >> data[i][j];
      
      if(target)
        f >> y[i];
    }
  }

   *data_ = data;
   *y_ = y;
}

void OldIOTorch::loadData(string file, sreal ***data_, real **y_,
                       int &l, int &w, bool bin, bool target)
{
  sreal **data = NULL;
  real *y = NULL;
  int *indices;
  real *valeurs;
  FILE *f;

  if(bin)
  {
    if( (f = fopen(file.c_str(), "rb")) == NULL)
      error();
    
    fread(&l, sizeof(int), 1, f);
    fread(&w, sizeof(int), 1, f);
    
    indices = new int[w];
    valeurs = new real[w];
    data = new sreal*[l];
    if(target)
      y = new real[l];

    for(int i = 0; i < l; i++)
    {
      int j = 0;
      int i_lu;
      while(1)
      {
        fread(&i_lu, sizeof(int), 1, f);
        if(i_lu < 0)
          break;
			
        indices[j] = i_lu;
        fread(&valeurs[j], sizeof(real), 1, f);
        j++;
      }

      /*** Fill it ***/
      data[i] = new sreal[j+1];
      data[i][j].indice = -1;

      for(int k = 0; k < j; k++)
      {
        data[i][k].indice = indices[k];
        data[i][k].valeur = valeurs[k];
      }
    }

    if(target)
      fread(y, sizeof(real), l, f);

    fclose(f);
  }
  else
  {
    if( (f = fopen(file.c_str(), "rt")) == NULL)
      error();

    fscanf(f, "%d", &l);
    fscanf(f, "%d", &w);
    if(target)
      w -= 1;
    else  // Je sais foutrement plus pourquoi
    {
      /*** Search the end line ***/
      while(1)
      {
        char c;
        int ci;

        ci = getc(f);
        c = (char)ci;
        if( c == '\n')
          break;
      }      
    }

    indices = new int[w];
    valeurs = new real[w];
    data = new sreal*[l];
    if(target)
      y = new real[l];

    real f_lu;
  	char c;
    int ci;

    for(int i = 0; i < l; i++)
    {
      int j = 0;
      while(1)
      {
        if(target)
        {
#ifdef USEDOUBLE
          fscanf(f, "%lf", &f_lu);
#else
          fscanf(f, "%f", &f_lu);
#endif
        }

        while(1)
        {
          ci = getc(f);
          c = (char)ci;
          if( c == '\n')
            break;

          if(target)
          {
            if( (c == '.') || (c == '-') || ( (c >= '0') && (c <= '9') ) )
            {
              ungetc(ci, f);
              break;
            }
          }
          else // Ouais ?!. C'etait le bordel monstre ce truc.
          {
            if( (c >= '0') && (c <= '9') )
            {
              ungetc(ci, f);
              break;
            }
          }
        }
        
        if(c == '\n')
          break;

        if(target)
          indices[j] = (int)f_lu;
        else
        {
          fscanf(f, "%d", &indices[j]);
        }

#ifdef USEDOUBLE
        fscanf(f, "%lf", &valeurs[j]);
#else
        fscanf(f, "%f", &valeurs[j]);
#endif
        j++;
      }

      if(target)
        y[i] = f_lu;
      data[i] = new sreal[j+1];
      data[i][j].indice = -1;

      for(int k = 0; k < j; k++)
      {
        data[i][k].indice = indices[k];
        data[i][k].valeur = valeurs[k];
      }
    }
  }

  delete[] indices;
  delete[] valeurs;

  *data_ = data;
  *y_ = y;
}

