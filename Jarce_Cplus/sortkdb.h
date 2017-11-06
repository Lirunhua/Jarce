/* 
 * File:   sortkdb.h
 * Author: Administrator
 *
 * Created on 2016年12月3日, 下午9:43
 */
#pragma once

#include <limits.h>

#include "incrementalLearner.h"
#include "distributionTree.h"
#include "xxxyDist.h"
#include "xxxxyDist.h"
#include "yDist.h"

class sortkdb : public IncrementalLearner {
public:
    sortkdb();
    sortkdb(char*const*& argv, char*const* end);
    ~sortkdb(void);

    void reset(InstanceStream &is); ///< reset the learner prior to training
    void initialisePass(); ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
    void train(const instance &inst); ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
    void finalisePass(); ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
    bool trainingIsFinished(); ///< true iff no more passes are required. updated by finalisePass()
    void getCapabilities(capabilities &c);

    virtual void classify(const instance &inst, std::vector<double> &classDist);
    //反馈
    void getNoCatAtts_(unsigned int &NoCatAtt);
    void getStructure(std::vector<std::vector<CategoricalAttribute> > &parents,std::vector<CategoricalAttribute> &order);
    void chang_parents(std::vector<std::vector<CategoricalAttribute> > &parents_change);
    virtual void classify_change(const instance& inst, std::vector<double> &posteriorDist, std::vector<std::vector<CategoricalAttribute> > &parents_);
    
    
protected:
    unsigned int pass_; ///< the number of passes for the learner
    unsigned int k_; ///< the maximum number of parents
    unsigned int noCatAtts_; ///< the number of categorical attributes.
    unsigned int noClasses_; ///< the number of classes
    bool union_localkdb;   //加局部kdb模型
    
    
    double sum_unmi;
    double sum_uncmi;
    
    xxxyDist dist_; // used in the first pass
    xxxyDist dist_1;
    yDist classDist_; // used in the second pass and for classification
    //std::vector<distributionTree> dTree_;                      // used in the second pass and for classification
    std::vector<CategoricalAttribute> order_;
    std::vector<std::vector<CategoricalAttribute> > parents_;
    std::vector<std::vector<CategoricalAttribute> > parents_1;
    InstanceStream* instanceStream_;
    bool trainingIsFinished_;
};
