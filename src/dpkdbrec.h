/* 
 * File:   dpkdbrec.h
 * Author: Administrator
 *
 * Created on 2016年11月29日, 下午2:54
 */
#pragma once

#include <limits.h>

#include "incrementalLearner.h"
#include "distributionTree.h"
#include "xxxyDist.h"
#include "xxxxyDist.h"
#include "yDist.h"

class dpkdbrec : public IncrementalLearner {
public:
    dpkdbrec();
    dpkdbrec(char*const*& argv, char*const* end);
    ~dpkdbrec(void);

    void reset(InstanceStream &is); ///< reset the learner prior to training
    void initialisePass(); ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
    void train(const instance &inst); ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
    void finalisePass(); ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
    bool trainingIsFinished(); ///< true iff no more passes are required. updated by finalisePass()
    void getCapabilities(capabilities &c);

    virtual void classify(const instance &inst, std::vector<double> &classDist);
    void displayInfo(const instance& inst, std::vector<std::vector<CategoricalAttribute> > parents_);
protected:
    unsigned int pass_; ///< the number of passes for the learner
    unsigned int k_; ///< the maximum number of parents
    unsigned int noCatAtts_; ///< the number of categorical attributes.
    unsigned int noClasses_; ///< the number of classes
    unsigned int arc;

    unsigned int arc_sum;
    unsigned int arc_loc_sum;
    unsigned int arc_same_sum;
    double arc_ratio;
    double wer;
    double differ;
    double posteriorDist_diff;
    unsigned int hunhe_correct_num;
    unsigned int hunhe_correct_num_bydiff;

    unsigned int hunhe_wrong_num;
    unsigned int diff_num;
    unsigned int kdb_correct_num;
    unsigned int kdb_wrong_num;

    double hubulv;  
    double wupanlv;
    double jiucuolv;


    bool subsumptionResolution; ///<true if selecting active parents for each instance，优化则加参数-sub
    bool add_localkdb; //是否构造局部模型 ，构造则加参数-loc
    bool union_kdb_localkdb; //是否计算全局kdb的联合概率，并与局部联合概率结合  -un ，前提add_localkdb=true，即有参数-loc
    unsigned int minCount;
    unsigned int fathercount[100];


    xxxyDist dist_; // used in the first pass
    xxxyDist dist_1; //k=2
    yDist classDist_; // used in the second pass and for classification
    //std::vector<distributionTree> dTree_;                      // used in the second pass and for classification
    std::vector<std::vector<CategoricalAttribute> > parents_;
    std::vector<std::vector<CategoricalAttribute> > parents_0;
    std::vector<CategoricalAttribute> order_;
    std::vector<CategoricalAttribute> order_loc;
    std::vector<std::vector<CategoricalAttribute> > parents_1;
    std::vector<std::vector<CategoricalAttribute> > parents_2;
    std::vector<std::vector<CategoricalAttribute> > parents_3;
    bool trainingIsFinished_;
    InstanceStream* instanceStream_;
};

