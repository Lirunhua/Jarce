/* Open source system for classification learning from very large data
** Class for creating table of mutual information between the class and each Attribute from an xyDist
**
** Copyright (C) 2012 Geoffrey I Webb
**
** This program is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
** 
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
** GNU General Public License for more details.
** 
** You should have received a copy of the GNU General Public License
** along with this program. If not, see <http://www.gnu.org/licenses/>.
**
** Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
*/
#pragma once
#include "xyDist.h"
#include <stdio.h>

class classAttMI
{
public:
  classAttMI(xyDist &dist);
  ~classAttMI(void);

  inline float &operator[](const CategoricalAttribute i) {
    return mi[i];
  }

  // used for comparing two attributes relative to mi
  bool gt(CategoricalAttribute i, CategoricalAttribute j) const { return (mi[i]>mi[j]);}

  void print() {
    for (std::vector<float>::const_iterator it = mi.begin(); it != mi.end(); it++) {
      printf("%6.3f ", *it);
    }
    putchar('\n');
  }

  std::vector<float> mi;
};
