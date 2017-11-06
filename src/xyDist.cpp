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
#include "xyDist.h"
#include "utils.h"

#include <memory.h>

xyDist::xyDist() {
}

xyDist::xyDist(InstanceStream *is)
{
  /*调用了自己的函数*/
  reset(is);

  instance inst(*is);

  while (!is->isAtEnd()) {
    if (is->advance(inst))
        /*调用了自己的函数*/
        update(inst);
  }
}

xyDist::~xyDist(void)
{
}

/*
 * 主要是初始化空间，初始化counts_,和classCounts
 * 初始时赋值均为0
 */
void xyDist::reset(InstanceStream *is) {

  metaData_ = is->getMetaData();
  /*返回的是类取值的数目*/
  noOfClasses_ = is->getNoClasses();
  /*count用于标记实例*/
  count = 0;
  
  /*
   * 为存储所有可能的情况分配空间
   * 每一个单位存储属性和类对应实例的数目
   */
  
  counts_.resize(is->getNoCatAtts());/*is->getNoCatAtts返回属性的数目*/

  for (CategoricalAttribute a = 0; a < is->getNoCatAtts(); a++) {
    /*is->getNoValues(a)返回每种属性肯能取值的数目*/
    counts_[a].assign(is->getNoValues(a)*noOfClasses_, 0);
  }
  
  classCounts.assign(noOfClasses_, 0);
}

void xyDist::update(const instance &inst) {
  count++;
  /*得到该实例的类取值，并统计到classCounts中，
   这里面所有的属性，属性值，类值，都泛化为数字*/
  const CatValue y = inst.getClass();
  classCounts[y]++;

  for (CategoricalAttribute a = 0; a < metaData_->getNoCatAtts(); a++) {
    counts_[a][inst.getCatVal(a)*noOfClasses_+y]++;
  }
}

void xyDist::clear(){
  /*the size of classCounts is 0*/
  classCounts.clear();
  /*counts_的空间没变，只是每个变量的值变为0*/
  for (CategoricalAttribute a = 0; a < getNoAtts(); a++) {
    counts_[a].assign(metaData_->getNoValues(a)*noOfClasses_, 0);
  }
}
