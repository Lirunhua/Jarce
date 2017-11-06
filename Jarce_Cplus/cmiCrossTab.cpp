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
#include "cmiCrossTab.h"
#include "smoothing.h"
#include "utils.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h> // required for abs?


cmiCrossTab::cmiCrossTab(xxyDist &dist, const bool discrVals) : crosstab<float>(dist.xyCounts.getNoCatAtts())
{
  const double totalCount = dist.xyCounts.count;

  for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
      for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
      double m = 0.0;

      if (discrVals) {
        std::vector<double> classDistIndep(dist.getNoClasses());
        std::vector<double> classDist(dist.getNoClasses());

        for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
          for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
            for (CatValue y = 0; y < dist.getNoClasses(); y++) {
              const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
              if (x1x2y) {
                classDistIndep[y] = dist.xyCounts.p(y) * dist.xyCounts.p(x1, v1, y) * dist.xyCounts.p(x2, v2, y);
                classDist[y] = dist.xyCounts.p(y) * dist.xyCounts.p(x1, v1, y) * dist.p(x2, v2, x1, v1, y);
              }
              else {
                classDistIndep[y] = 0.0;
                classDist[y] = 0.0;
              }
            }
            normalise(classDistIndep);
            normalise(classDist);
            for (CatValue y = 0; y < dist.getNoClasses(); y++) {
              m += abs(classDistIndep[y] - classDist[y]);
            }
          }
        }
      }
      else {
        for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
          for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
            for (CatValue y = 0; y < dist.getNoClasses(); y++) {
              const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
              if (x1x2y) {
                //const unsigned int yCount = dist->xyCounts.getClassCount(y);
                //const unsigned int  x1y = dist->xyCounts.getCount(x1, v1, y);
                //const unsigned int  x2y = dist->xyCounts.getCount(x2, v2, y);
                m += (x1x2y/totalCount) * log2(dist.xyCounts.getClassCount(y) * x1x2y / (static_cast<double>(dist.xyCounts.getCount(x1, v1, y))*dist.xyCounts.getCount(x2, v2, y)));
              }
            }
          }
        }

        assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision
      }

      (*this)[x1][x2] = m;
      (*this)[x2][x1] = m;
    }
  }
}

cmiCrossTab::~cmiCrossTab(void)
{
}
