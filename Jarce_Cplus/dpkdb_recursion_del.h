/* 
 * File:   dpkdb_recursion_del.h
 * Author: Administrator
 *
 * Created on 2016年8月30日, 下午1:08
 */

#pragma once

#include <limits.h>

#include "incrementalLearner.h"
#include "distributionTree.h"
#include "xxxyDist.h"
#include "xxxxyDist.h"
#include "yDist.h"

/**
<!-- globalinfo-start -->
 * Class for a k-dependence Bayesian classifier.<br/>
 * <br/>
 * For more information on k-dependence Bayesian classifiers, see:<br/>
 * <br/>
 * Sahami, M.: Learning limited dependence Bayesian classifiers. In: KDD-96: 
 * Proceedings of the Second International Conference on Knowledge Discovery and
 * Data Mining, 335--338, 1996.
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * \@inproceedings{sahami1996learning,
 *   title={Learning limited dependence Bayesian classifiers},
 *   author={Sahami, M.},
 *   booktitle={KDD-96: Proceedings of the Second International Conference on 
 *              Knowledge Discovery and Data Mining},
 *   pages={335--338},
 *   year={1996}
 * }
 * </pre>
 <!-- technical-bibtex-end -->
 *
 *
 * @author Geoff Webb (geoff.webb@monash.edu)
 */


class dpkdb_recursion_del : public IncrementalLearner {
public:
    dpkdb_recursion_del();
    dpkdb_recursion_del(char*const*& argv, char*const* end);
    ~dpkdb_recursion_del(void);

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
    unsigned int arc;
    xxxyDist dist_; // used in the first pass
    xxxyDist dist_1;  //k=2
    
    yDist classDist_; // used in the second pass and for classification
    //std::vector<distributionTree> dTree_;                      // used in the second pass and for classification
    std::vector<std::vector<CategoricalAttribute> > parents_;
    std::vector<std::vector<CategoricalAttribute> > parents_0;
    std::vector<CategoricalAttribute> order_;
    std::vector<std::vector<CategoricalAttribute> > parents_1;
    std::vector<std::vector<CategoricalAttribute> > parents_2;
    std::vector<std::vector<CategoricalAttribute> > parents_3;
    bool trainingIsFinished_;
    InstanceStream* instanceStream_;
};


