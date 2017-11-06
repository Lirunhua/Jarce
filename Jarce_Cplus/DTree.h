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
#include "learner.h"
#include "IndirectInstanceStream.h"
#include "xyDist.h"
#include <vector>
#include <set>
#include <limits>

class DTree;

class DTTrainingParams {
public:
  DTTrainingParams(const unsigned int minLeafSize, const float m) : minLeafSize_(minLeafSize), m_(m) {}

  virtual void makeMePolymorphic() {}

  unsigned int minLeafSize_;  ///< the minimum size of a node before we consider branching on it
  double m_;                   ///< the value of m for m-estimation
};

class DTClassificationParams {
public:
  DTClassificationParams(const double m) : m_(m) {}

  double m_; ///< value of m for m-estimate
};

class DTNode {
public:
  virtual ~DTNode() {}

  virtual void classify(const instance &inst, std::vector<double> &classDist) const = 0;  ///< classify the instance returning the class distribution
  virtual void print(const unsigned int depth, InstanceStream &source) const  = 0;        ///< print the (sub)tree
  virtual bool isEmpty() const  = 0;                                                     ///< true iff the (sub)tree has no training instances

};

class DTCatNode : public DTNode {
public:
  DTCatNode(const CategoricalAttribute att,
            AddressableInstanceStream &instances,
            DTTrainingParams *params,
            std::set<CategoricalAttribute> &blockedCatAtts,
            std::set<CategoricalAttribute> &blockedNumAtts,
            const std::vector<float> &priors,
            DTNode* (*growTree)(IndirectInstanceStream &instances,
                                DTTrainingParams *params,
                                std::set<CategoricalAttribute> &blockedCatAtts,
                                std::set<NumericAttribute> &blockedNumAtts,
                                const std::vector<float> &priors
                        ) );                                                  ///< create a new subtree rooted in a test on att
  ~DTCatNode();

  void classify(const instance &inst, std::vector<double> &classDist) const; ///< classify the instance returning the class distribution
  void print(const unsigned int depth, InstanceStream &source) const;        ///< print the (sub)tree
  bool isEmpty() const;                                                      ///< true iff the (sub)tree has no training instances

  CategoricalAttribute att_;
  std::vector<DTNode*> branches_;
};

class DTNumNode : public DTNode {
public:
  DTNumNode(const CategoricalAttribute att,
            const NumValue cut,
            AddressableInstanceStream &instances,
            DTTrainingParams *params,
            std::set<CategoricalAttribute> &blockedCatAtts,
            std::set<CategoricalAttribute> &blockedNumAtts,
            const std::vector<float> &priors,
            DTNode* (*growTree)(IndirectInstanceStream &instances,
                                DTTrainingParams *params,
                                std::set<CategoricalAttribute> &blockedCatAtts,
                                std::set<NumericAttribute> &blockedNumAtts,
                                const std::vector<float> &priors
                        )); ///< create a new subtree rooted in a test on att
  ~DTNumNode();

  void classify(const instance &inst, std::vector<double> &classDist) const; ///< classify the instance returning the class distribution
  void print(const unsigned int depth, InstanceStream &source) const;        ///< print the (sub)tree
  bool isEmpty() const;                                                      ///< true iff the (sub)tree has no training instances

  const NumericAttribute att_;
  const NumValue cut_;
  std::vector<DTNode*> branches_;
};

class DTLeaf : public DTNode {
public:
  DTLeaf(const std::vector<float> distribution, const InstanceCount count); ///< create a new leaf
  ~DTLeaf();
  
  void classify(const instance &inst, std::vector<double> &classDist) const; ///< classify the instance returning the class distribution
  void print(const unsigned int depth, InstanceStream &source) const;        ///< print the (sub)tree
  bool isEmpty() const;                                                      ///< true iff the (sub)tree has no training instances

  std::vector<float> distribution_;
  InstanceCount count_;
};

class DTree : public learner {
public:
  DTree(char*const*& argv, char*const* end);
  ~DTree(void);

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

  DTNode *root_;
  InstanceStream* source_;
  DTTrainingParams trainingParams_;
  DTClassificationParams classificationParams_;
  std::vector<double> priors; ///< the prior probability of each class
  bool trainingIsFinished_;
};

bool atLeastTwoValsAreSufficientlyFrequent(xyDist& dist, const CategoricalAttribute a, InstanceCount minfreq);

