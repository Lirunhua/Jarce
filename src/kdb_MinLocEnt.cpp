/* 
 * File:   kdb_MinLocEnt.cpp
 * Author: Administrator
 * 
 * Created on 2016年9月5日, 上午10:11
 */

#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>

#include "kdb_MinLocEnt.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

kdb_MinLocEnt::kdb_MinLocEnt() : pass_(1) {
}

kdb_MinLocEnt::kdb_MinLocEnt(char*const*& argv, char*const* end) : pass_(1) {
    name_ = "kdb_MinLocEnt";

    // defaults
    k_ = 1;

    // get arguments
    while (argv != end) {
        if (*argv[0] != '-') {
            break;
        } else if (argv[0][1] == 'k') {
            getUIntFromStr(argv[0] + 2, k_, "k");
        } else {
            break;
        }

        name_ += argv[0];

        ++argv;
    }
}

kdb_MinLocEnt::~kdb_MinLocEnt(void) {
}

void kdb_MinLocEnt::getCapabilities(capabilities &c) {
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

// creates a comparator for two attributes based on their relative mutual information with the class

class miCmpClass {
public:

    miCmpClass(std::vector<float> *m) {
        mi = m;
    }

    bool operator()(CategoricalAttribute a, CategoricalAttribute b) {
        return (*mi)[a] > (*mi)[b];
    }

private:
    std::vector<float> *mi;
};

class hmiCmpClass {
public:

    hmiCmpClass(std::vector<float> *m) {
        hmi = m;
    }

    bool operator()(CategoricalAttribute a, CategoricalAttribute b) {
        return (*hmi)[a] < (*hmi)[b];
    }

private:
    std::vector<float> *hmi;
};

void kdb_MinLocEnt::reset(InstanceStream &is) {
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();

    k_ = min(k_, noCatAtts_ - 1); // k cannot exceed the real number of categorical attributes - 1

    // initialise distributions
    //dTree_.resize(noCatAtts);
    parents_.resize(noCatAtts);

    for (CategoricalAttribute a = 0; a < noCatAtts; a++) {
        parents_[a].clear();
        //dTree_[a].init(is, a);
    }

    /*初始化各数据结构空间*/
    dist_.reset(is); //
    dist_1.reset(is); //
    classDist_.reset(is);

    pass_ = 1;
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass

/*通过训练集来填写数据空间*/
void kdb_MinLocEnt::train(const instance &inst) {
    if (pass_ == 1) {
        dist_.update(inst);
        dist_1.update(inst);
    } else {
        assert(pass_ == 2);


        classDist_.update(inst);
    }
}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void kdb_MinLocEnt::initialisePass() {
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void kdb_MinLocEnt::finalisePass() {
    if (pass_ == 1 && k_ != 0) {
        // calculate the mutual information from the xy distribution
        std::vector<float> mi;
        getMutualInformation(dist_.xxyCounts.xyCounts, mi);

        if (verbosity >= 3) {
            printf("\nMutual information table\n");
            print(mi);
        }

        // calculate the conditional mutual information from the xxy distribution
        crosstab<float> cmi = crosstab<float>(noCatAtts_);
        getCondMutualInf(dist_.xxyCounts, cmi);

        dist_.clear();

        if (verbosity >= 3) {
            printf("\nConditional mutual information table\n");
            cmi.print();
        }

        // sort the attributes on MI with the class
        std::vector<CategoricalAttribute> order;

        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            order.push_back(a);
        }

        // assign the parents
        if (!order.empty()) {
            miCmpClass cmp(&mi);

            std::sort(order.begin(), order.end(), cmp);

            if (verbosity >= 2) {
                printf("\n%s parents:\n", instanceStream_->getCatAttName(order[0]));
            }

            // proper KDB assignment of parents
            for (std::vector<CategoricalAttribute>::const_iterator it = order.begin() + 1; it != order.end(); it++) {
                parents_[*it].push_back(order[0]);
                for (std::vector<CategoricalAttribute>::const_iterator it2 = order.begin() + 1; it2 != it; it2++) {
                    // make parents into the top k attributes on mi that precede *it in order
                    if (parents_[*it].size() < k_) {
                        // create space for another parent
                        // set it initially to the new parent.
                        // if there is a lower value parent, the new parent will be inserted earlier and this value will get overwritten
                        parents_[*it].push_back(*it2);
                    }
                    for (unsigned int i = 0; i < parents_[*it].size(); i++) {
                        if (cmi[*it2][*it] > cmi[parents_[*it][i]][*it]) {
                            // move lower value parents down in order
                            for (unsigned int j = parents_[*it].size() - 1; j > i; j--) {
                                parents_[*it][j] = parents_[*it][j - 1];
                            }
                            // insert the new att
                            parents_[*it][i] = *it2;
                            break;
                        }
                    }
                }

                if (verbosity >= 2) {
                    printf("%s parents: ", instanceStream_->getCatAttName(*it));
                    for (unsigned int i = 0; i < parents_[*it].size(); i++) {
                        printf("%s ", instanceStream_->getCatAttName(parents_[*it][i]));
                    }
                    putchar('\n');
                }
            }
        }
    }

    ++pass_;
}

/// true iff no more passes are required. updated by finalisePass()

bool kdb_MinLocEnt::trainingIsFinished() {
    return pass_ > 2;
}

void kdb_MinLocEnt::classify(const instance& inst, std::vector<double> &posteriorDist) {

    parents_1.resize(noCatAtts_);
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        parents_1[a].clear();
    }
    parents_2.resize(noCatAtts_);
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        parents_2[a].clear();
    }

    std::vector<float> hmi;
    getlocH_xy(dist_1.xxyCounts.xyCounts, hmi, inst);

    std::vector<CategoricalAttribute> horder;
    horder.clear();
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        horder.push_back(a);
    }

    if (!horder.empty()) {
        hmiCmpClass hcmp(&hmi);
        std::sort(horder.begin(), horder.end(), hcmp);
        //        printf("horder2:\n");
        //        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        //            printf("%d\t",horder[a]);
        //        }printf("\n");
        std::vector<CategoricalAttribute> temp;
        temp.clear();
        double H = 0.0;
        for (std::vector<CategoricalAttribute>::const_iterator it = horder.begin(); it != horder.end(); it++) {
            printf("%d\n",it);
        }printf("\n");



    }
    double H_new1 = 0.0;
    //H_new1 = H_standard_loc_k2(classDist_, dist_1, parents_1, inst);
    //printf("************%lf\n", H_new1);
    //printf("--------------------------------------------------------------------\n");
    

//    printf("parents_1:\n");
//    for (unsigned int i = 0; i < noCatAtts_; i++) {
//        if (parents_1[i].size() == 1) {
//            printf("parents1[%d][0]=%d\n", i, parents_1[i][0]);
//        }
//        if (parents_1[i].size() > 1) {
//            printf("parents1[%d][0]=%d\tparents1[%d][1]=%d\n", i, parents_1[i][0], i, parents_1[i][1]);
//        }
//    }
    
    
    
    
    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);

    }
    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        for (CatValue y = 0; y < noClasses_; y++) {
            if (parents_1[x1].size() == 0) {
                // printf("PARent=0  \n");

                posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate


            } else if (parents_1[x1].size() == 1) {
                //  printf("PARent=1  \n");
                const InstanceCount totalCount1 = dist_1.xxyCounts.xyCounts.getCount(parents_1[x1][0], inst.getCatVal(parents_1[x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_1[x1][0], inst.getCatVal(parents_1[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents_1[x1].size() == 2) {
                // printf("PARent=2  \n");
                const InstanceCount totalCount1 = dist_1.xxyCounts.getCount(parents_1[x1][0], inst.getCatVal(parents_1[x1][0]), parents_1[x1][1], inst.getCatVal(parents_1[x1][1]));
                if (totalCount1 == 0) {
                    const InstanceCount totalCount2 = dist_1.xxyCounts.xyCounts.getCount(parents_1[x1][0], inst.getCatVal(parents_1[x1][0]));
                    if (totalCount2 == 0) {
                        posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_1[x1][0], inst.getCatVal(parents_1[x1][0]), y);
                    }
                } else {
                    posteriorDist[y] *= dist_1.p(x1, inst.getCatVal(x1), parents_1[x1][0], inst.getCatVal(parents_1[x1][0]), parents_1[x1][1], inst.getCatVal(parents_1[x1][1]), y);
                }

            }
        }
    }
    // normalise the results
    normalise(posteriorDist);
}