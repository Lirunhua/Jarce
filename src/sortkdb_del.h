/* 
 * File:   sortkdb_del.h
 * Author: Administrator
 *
 * Created on 2016年12月14日, 上午10:18
 */
#pragma once

#include <limits.h>

#include "incrementalLearner.h"
#include "distributionTree.h"
#include "xxxyDist.h"
#include "xxxxyDist.h"
#include "yDist.h"

class sortkdb_del : public IncrementalLearner {
public:
    sortkdb_del();
    sortkdb_del(char*const*& argv, char*const* end);
    ~sortkdb_del(void);

    void reset(InstanceStream &is); ///< reset the learner prior to training
    void initialisePass(); ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
    void train(const instance &inst); ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
    void finalisePass(); ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
    bool trainingIsFinished(); ///< true iff no more passes are required. updated by finalisePass()
    void getCapabilities(capabilities &c);

    virtual void classify(const instance &inst, std::vector<double> &classDist);
   
    
protected:
    unsigned int pass_; ///< the number of passes for the learner
    unsigned int k_; ///< the maximum number of parents
    unsigned int noCatAtts_; ///< the number of categorical attributes.
    unsigned int noClasses_; ///< the number of classes
    
    
    xxxyDist dist_; // used in the first pass
    xxxyDist dist_1;
    yDist classDist_; // used in the second pass and for classification
    //std::vector<distributionTree> dTree_;                      // used in the second pass and for classification
    std::vector<CategoricalAttribute> order_;
    std::vector<bool> del;
    std::vector<std::vector<CategoricalAttribute> > parents_;
    InstanceStream* instanceStream_;
    bool trainingIsFinished_;
};
