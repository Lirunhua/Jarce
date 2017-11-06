/* Open source system for classification learning from very large data
** Class for a decision tree learner
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
#include "DTree.h"
#include <vector>
#include <set>
#include <limits>


class RFDTTrainingParams : public DTTrainingParams {
public:
  RFDTTrainingParams(const unsigned int minLeafSize, const float m, const unsigned int n) : DTTrainingParams(minLeafSize, m), noOfAttsConsidered_(n) {}

  unsigned int noOfAttsConsidered_; ///< the number of randomly selected attributes to consider at each node
};

class RFDTree : public DTree {
public:
  RFDTree(char*const*& argv, char*const* end);
  ~RFDTree(void);

  // learner methods
  void train(InstanceStream &is);       ///< train the classifier from an instance stream
  void classify(const instance &inst, std::vector<double> &classDist);  ///< infer the class distribution for the current instance in the instance stream
  void getCapabilities(capabilities &c); ///< describes what kind of data the learner is able to handle

  void printClassifier();       ///< print details of the classifier that has been created

  /// grow a (sub)tree.
  /// make this static so that it can be called without needing access to the root of the tree
  static DTNode* growTree(IndirectInstanceStream &instances,
                          DTTrainingParams *params,
                          std::set<CategoricalAttribute> &blockedCatAtts,
                          std::set<NumericAttribute> &blockedNumAtts,
                          const std::vector<float> &priors);

  RFDTTrainingParams trainingParams_;
};

