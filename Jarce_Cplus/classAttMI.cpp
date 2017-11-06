/* Open source system for classification learning from very large data
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
#include "classAttMI.h"
#include "utils.h"
#include <math.h>

classAttMI::classAttMI(xyDist &dist)
{
  mi.assign(dist.getNoCatAtts(), 0.0);

  const double totalCount = dist.count;

  for (CategoricalAttribute a = 0; a < dist.getNoCatAtts(); a++) {
    double m = 0.0;

    for (CatValue v = 0; v < dist.getNoValues(a); v++) {
      for (CatValue y = 0; y < dist.getNoClasses(); y++) {
        const InstanceCount avyCount = dist.getCount(a,v,y);
        if (avyCount) {
          m += (avyCount / totalCount) * log2(avyCount/((dist.getCount(a, v)/totalCount) * dist.getClassCount(y)));
        }
      }
    }

    mi[a] = m;
  }
}

classAttMI::~classAttMI(void)
{
}
