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

#include <assert.h>
#include <float.h>
#include <stdlib.h>

#include "utils.h"
#include "nb.h"


nb::nb(char*const*& argv, char*const* end) : xyDist_(), trainingIsFinished_(false)
{ 
  name_ = "Naive Bayes";
}


nb::~nb(void)
{
}

void  nb::getCapabilities(capabilities &c){
  c.setCatAtts(true);  // only categorical attributes are supported at the moment
}

void nb::reset(InstanceStream &is) {
  /*初始数据结构空间*/
  xyDist_.reset(&is);
  trainingIsFinished_ = false;
}


void nb::train(const instance &inst) {
  /*进行数据统计*/
  xyDist_.update(inst);
}


void nb::initialisePass() {
}


void nb::finalisePass() {
  trainingIsFinished_ = true;
}


bool nb::trainingIsFinished() {
  return trainingIsFinished_;
}

/*
 * 该函数是每次单独对一个实例进行分类
 */
void nb::classify(const instance &inst, std::vector<double> &classDist) {
  const unsigned int noClasses = xyDist_.getNoClasses();

  for (CatValue y = 0; y < noClasses; y++) {
    double p = xyDist_.p(y) * (std::numeric_limits<double>::max() / 2.0); 
    // scale up by maximum possible factor to reduce risk of numeric underflow

    for (CategoricalAttribute a = 0; a < xyDist_.getNoAtts(); a++) {
        p *= xyDist_.p(a, inst.getCatVal(a), y);
    }
    /*最后求得p就是c(inst)*/

    assert(p >= 0.0);
    /*classDist[y]存储的是实例属于每种类值的概率*/
    classDist[y] = p;
  }

  /*标准化类值概率，不影响类值概率大小关系*/
  normalise(classDist);
}



