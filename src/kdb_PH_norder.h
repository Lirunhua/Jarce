/* 
 * File:   kdb_PH_norder.h
 * Author: Administrator
 *
 * Created on 2016年9月26日, 上午9:57
 */

#pragma once

#include <limits.h>

#include "incrementalLearner.h"
#include "distributionTree.h"
#include "xxyDist.h"
#include "xxxyDist.h"
#include "yDist.h"


class kdb_PH_norder :  public IncrementalLearner
{
public:
  kdb_PH_norder();
  kdb_PH_norder(char*const*& argv, char*const* end);
  ~kdb_PH_norder(void);

  void reset(InstanceStream &is);   ///< reset the learner prior to training
  void initialisePass();            ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
  void train(const instance &inst); ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
  void finalisePass();              ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
  bool trainingIsFinished();        ///< true iff no more passes are required. updated by finalisePass()
  void getCapabilities(capabilities &c);

  virtual void classify(const instance &inst, std::vector<double> &classDist);

protected:
  unsigned int pass_;                                        ///< the number of passes for the learner
  unsigned int k_;                                           ///< the maximum number of parents
  unsigned int noCatAtts_;                                   ///< the number of categorical attributes.
  unsigned int noClasses_;                                   ///< the number of classes
  unsigned int kdb_root;
  xxyDist dist_;                                             // used in the first pass
  xxxyDist dist_1; 
  yDist classDist_;                                          // used in the second pass and for classification
  yDist classDist_1;   
  std::vector<CategoricalAttribute> order_;
  std::vector<CategoricalAttribute> order_mi;
  //std::vector<distributionTree> dTree_;                      // used in the second pass and for classification
  //std::vector<distributionTree> dTree_1;  
  std::vector<std::vector<CategoricalAttribute> > parents_;
  std::vector<std::vector<CategoricalAttribute> > parents_1;
  std::vector<std::vector<CategoricalAttribute> > parents_2;
  //std::vector<std::vector<CategoricalAttribute> > parents_2;
  std::vector<std::vector<CategoricalAttribute> > parents_temp;
  bool trainingIsFinished_;
  InstanceStream* instanceStream_;
};
