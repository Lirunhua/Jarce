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

#include "kdb.h"
#include "xxxyDist.h"

/**
<!-- globalinfo-start -->
 * Class for different variations of a k-dependence Bayesian classifier.<br/>
 <!-- globalinfo-end -->
 *
 * @author Geoff Webb (geoff.webb@monash.edu)
 * @author Ana M. Martinez (anam.martinez@monash.edu)
 */

class kdbExt : public kdb
{
public:
  kdbExt(char*const*& argv, char*const* end);
  ~kdbExt(void);

  void reset(InstanceStream &is);   
  void initialisePass(const int pass);
  virtual void train(const instance &inst);
  virtual void finalisePass();
  bool trainingIsFinished();        
  void getCapabilities(capabilities &c);
  
  virtual void classify(const instance &inst, std::vector<double> &classDist);

  void printClassifier();

private:
  bool tan_;                 ///< true if using KDB to implement TAN
  bool chisq_;               ///< true if attributes must pass a chi-squared test of independence with class
  bool holm_;                ///< true if holm-bonferroni adjustment is used with chisq tests
  bool chisqParents_;        ///< true if parents must pass a chi-squared test of independence with child
  unsigned int minCount_;    ///< minimum count required for probability estimation
  bool randomOrder_;         ///< true if using random order
  bool randomParents_;       ///< true if using random parents
  bool bestLinks_;           ///< true if parents asigned based on best links by cmi
  bool orderByValue_;        ///< true if using the total (sum) cmi  as the order
  bool discrimVals_;         ///< true if the error dif (dependece vs independence) is used instead of mi
  bool selective_;           ///< attribute selection using leave-one-out cross validation (loocv)
  bool selectiveTest_;       ///< attribute selection if significant difference (binomial test).
  bool selectiveWeighted_;   ///< use weighted RMSE for attribute selection.
  bool selectiveMCC_;        ///< used Matthews Correlation Coefficient for selection
  bool selectiveK_;          ///< selects the best k value
  bool selectiveSampling_;   ///< attribute selection using 100.000 samples 
  bool su_;                  ///< symmetrical uncertainty instead of mi and conditional symmetrical uncertainty instead of cmi
  bool orderBysu_;           ///< symmetrical uncertainty instead of mi, only for parents.
  bool orderBycmi_;          ///< select order based on the attributes included in the order so far
  bool cmiParents_;          ///< select order of parents based on the parents included in the order so far
  
  std::vector<bool> active_; ///< true for active[att] if att is selected -- flags: chisq, selective, selectiveTest
  unsigned int inactiveCnt_; ///< number of attributes not selected -- flags: chisq, selective, selectiveTest
  InstanceCount trainSize_;  ///< number of examples for training, to calculate the RMSE for each LOOCV -- flags: selective, selectiveTest
  std::vector<double> foldLossFunct_;  ///< loss function for every additional attribute (foldLossFunct_[noCatAtt]: only the class is considered) -- flags: selective, selectiveTest
  std::vector<std::vector<double> > foldLossFunctallK_;  ///< loss function for every additional attribute (foldSumRMSE[noCatAtt]: only the class is considered) for every k -- flags: selectiveK
  unsigned int bestK_;      ///< number of parents selected by kdb selective with selectiveK option
  std::vector<InstanceCount> binomialTestCounts_;  ///< number of wins compared to considering all the atts -- flags:  selectiveTest
  std::vector<InstanceCount> sampleSizeBinomTest_; ///< Sample size for a binomial test, leaving out ties -- flags: selectiveTest
//  std::vector<InstanceCount> TP_;  ///< Store TP count (needed for selectiveMCC)
//  std::vector<InstanceCount> FP_;  ///< Store FP count (needed for selectiveMCC)
//  std::vector<InstanceCount> TN_;  ///< Store TN count (needed for selectiveMCC)
//  std::vector<InstanceCount> FN_;  ///< Store FN count (needed for selectiveMCC)
  //std::vector<std::vector<InstanceCount> > TPallK_;  ///< Store TP count (needed for selectiveMCC with selectiveK)
  //std::vector<std::vector<InstanceCount> > FPallK_;  ///< Store FP count (needed for selectiveMCC with selectiveK)
  //std::vector<std::vector<InstanceCount> > TNallK_;  ///< Store TN count (needed for selectiveMCC with selectiveK)
  //std::vector<std::vector<InstanceCount> > FNallK_;  ///< Store FN count (needed for selectiveMCC with selectiveK)
  std::vector< std::vector< crosstab<InstanceCount> > > xtab_; ///< confusion matrix for all k values and all attributes (needed for selectiveMCC with selectiveK), only k=0 is used for plain selective
  xxxyDist xxxyDist_;        ///< xxxy distribution -- flags: orderBycmi, cmiParents and cmiAll
  std::vector<double> prior_;///< the prior for each class, saved at end of first pass -- flags: selectiveWeighted
  std::vector<CategoricalAttribute> order_;        ///< record the attributes in order based on different criteria
  unsigned int sampleSize_;                        ///< desired sample size -- flags: selectiveSampling
  unsigned int sampleSizeDec_;                     ///< sample size, used to count how many so far -- flags: selectiveSampling
  unsigned int trainSizeDec_;                      ///< training size, used to count how many remaining -- flags: selectiveSampling
  std::vector<bool> sampledInstaces;               ///< indexes of the instances to be considered for LOOCV -- flags: selectiveSampling
  MTRand_int32 rand;
};
