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
#pragma once
#include "instanceStream.h"
#include "utils.h"

const NumericAttribute NOPARENTEAGER = std::numeric_limits<NumericAttribute>::max();  // used because some compilers won't accept std::numeric_limits<NumericAttribute>::max() here

class dtNodeEager {
public:
  dtNodeEager();   // default constructor - init must be called after construction
  dtNodeEager(const CategoricalAttribute att);
  dtNodeEager(InstanceStream const* stream, const CategoricalAttribute att);
  ~dtNodeEager();

  inline void setStream(InstanceStream const* stream) { instanceStream_ = stream; }

  void init(InstanceStream const* stream, const CategoricalAttribute a);  // initialise a new uninitialised node
  void clear(CategoricalAttribute a);                          // reset a node to be empty

  // returns the start of the X=v,Y=y counts for value v
  inline InstanceCount &ref(const CatValue v, const CatValue y) {
    return xyCount.ref(v, y);
  }

  // returns the count X=v,Y=y
  inline InstanceCount getCount(const CatValue v, const CatValue y) {
    return xyCount.ref(v, y);
  }

  void updateStats(CategoricalAttribute target, std::vector<CategoricalAttribute> &parents, unsigned int k, unsigned int depthRemaining, unsigned long long int &pc, double &apd, unsigned long long int &zc);

  fdarray<double > condProbs;  // conditional probability indexed by x val the y val
  fdarray<InstanceCount> xyCount;  // joint count indexed by x val the y val
  ptrVec<dtNodeEager> children;
  CategoricalAttribute att;        // the Attribute whose values select the next child

private:
  static InstanceStream const *instanceStream_; // save just one metadata pointer for the whole tree
};

class distributionTreeEager
{
public:
  distributionTreeEager();   // default constructor - init must be called after construction
  distributionTreeEager(InstanceStream const* stream, const CategoricalAttribute att);
  ~distributionTreeEager(void);

  void init(InstanceStream const& stream, const CategoricalAttribute att);
  void clear(CategoricalAttribute a);                            // reset a tree to be empty

  void update(const instance &i, const CategoricalAttribute att, const std::vector<CategoricalAttribute> &parents, const unsigned int k);

  void calculateProbs(const CategoricalAttribute att);
  void findLeaf(dtNodeEager *dt,
			const CategoricalAttribute att);

  // update classDist using the evidence from the tree about i
  void updateClassDistribution(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i);  
  void updateClassDistribution(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i, InstanceCount minCount);
  //This method discounts i (Pazzani's trick for loocv)
  void updateClassDistributionloocv(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i);  

  // get statistics on the kdb structure.
  // pc = the number of paths defined by the parent structure = sum over atts of product over parents of number of values for parent 
  // apd = the average depth to which those paths are instantiated
  // zc = the number of counts in leaf nodes that are zero
  void updateStats(CategoricalAttribute target, std::vector<CategoricalAttribute> &parents, unsigned int k, unsigned long long int &pc, double &apd, unsigned long long int &zc);

private:
  dtNodeEager dTree;
  InstanceStream const* instanceStream_;
};
