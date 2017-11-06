/* 
 * File:   dpkdb_recursion_del.cpp
 * Author: Administrator
 * 
 * Created on 2016年8月30日, 下午1:08
 */

#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>

#include "dpkdb_recursion_del.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

dpkdb_recursion_del::dpkdb_recursion_del() : pass_(1) {
}

dpkdb_recursion_del::dpkdb_recursion_del(char*const*& argv, char*const* end) : pass_(1) {
    name_ = "dpkdb_recursion_del";

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

dpkdb_recursion_del::~dpkdb_recursion_del(void) {
}

void dpkdb_recursion_del::getCapabilities(capabilities &c) {
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

// creates a comparator for two attributes based on their relative mutual information with the class

class miCmpClass {
public:

    miCmpClass(std::vector<float> *m) {
        mi = m;
    }

    bool operator()(CategoricalAttribute a, CategoricalAttribute b) {
        return (*mi)[a] > (*mi)[b]; //mi
        //return (*mi)[a] < (*mi)[b]; //H(Y|xi)
    }

private:
    std::vector<float> *mi;
};

void dpkdb_recursion_del::reset(InstanceStream &is) {
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();
    arc = 0;
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

    trainingIsFinished_ = false;
    //pass_ = 1;
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass

/*通过训练集来填写数据空间*/
void dpkdb_recursion_del::train(const instance &inst) {
    dist_.update(inst);
    dist_1.update(inst);


    classDist_.update(inst);

}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void dpkdb_recursion_del::initialisePass() {
    assert(trainingIsFinished_ == false);
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void dpkdb_recursion_del::finalisePass() {
    assert(trainingIsFinished_ == false);
    // calculate the mutual information from the xy distribution
    std::vector<float> mi;
    //getH_xy(dist_.xxyCounts.xyCounts, mi);
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

        for (unsigned int i = 0; i < noCatAtts_; i++) {
            if (parents_[i].size() == 1) {
                arc++;
            }
            if (parents_[i].size() > 1) {
                arc += k_;
            }
        }
        //printf("arc_num=%d\n", arc);
//        printf("parents_:******************************\n");
//        for (unsigned int i = 0; i < noCatAtts_; i++) {
//            if (parents_[i].size() == 1) {
//                printf("parents[%d][0]=%d\n", i, parents_[i][0]);
//            }
//            if (parents_[i].size() > 1) {
//                printf("parents[%d][0]=%d\tparents[%d][1]=%d\n", i, parents_[i][0], i, parents_[i][1]);
//            }
//        }
    
    }

    trainingIsFinished_ = true;
}

/// true iff no more passes are required. updated by finalisePass()

bool dpkdb_recursion_del::trainingIsFinished() {
    return trainingIsFinished_;
}

void dpkdb_recursion_del::classify(const instance& inst, std::vector<double> &posteriorDist) {

    double H_standard = 0.0;
    H_standard = H_standard_loc_k2(classDist_, dist_1, parents_, inst);
    //printf("H_standard=%lf\n", H_standard);

    parents_1.resize(noCatAtts_);
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        parents_1[a].clear();
    }
    H_recursion_del(classDist_, dist_1, H_standard, arc, parents_, parents_1, inst);

//    printf("parents_1:\n");
//    for (unsigned int i = 0; i < noCatAtts_; i++) {
//        if (parents_1[i].size() == 1) {
//            printf("parents1[%d][0]=%d\n", i, parents_1[i][0]);
//        }
//        if (parents_1[i].size() > 1) {
//            printf("parents1[%d][0]=%d\tparents1[%d][1]=%d\n", i, parents_1[i][0], i, parents_1[i][1]);
//        }
//    }
//    printf("\n");

    
    //联合概率
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








