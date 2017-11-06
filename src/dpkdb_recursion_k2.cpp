/* 
 * File:   dpkdb_recursion_k2.cpp
 * Author: Administrator
 * 
 * Created on 2016年8月30日, 下午1:12
 */

#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>

#include "dpkdb_recursion_k2.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

dpkdb_recursion_k2::dpkdb_recursion_k2() : pass_(1) {
}

dpkdb_recursion_k2::dpkdb_recursion_k2(char*const*& argv, char*const* end) : pass_(1) {
    name_ = "dpkdb_recursion_k2";

    subsumptionResolution = false;
    add_localkdb = false;
    union_kdb_localkdb = false;
    minCount = 100;

    printf("!!\n");
    arc_sum = 0;
    arc_loc_sum = 0;
    arc_same_sum = 0;
    wer = 0;
    differ = 0.0;
    arc_ratio = 0.0;


    // defaults
    k_ = 1;

    // get arguments
    while (argv != end) {
        if (*argv[0] != '-') {
            break;
        } else if (argv[0][1] == 'k') {
            getUIntFromStr(argv[0] + 2, k_, "k");
        } else if (streq(argv[0] + 1, "sub")) {
            subsumptionResolution = true;
        } else if (streq(argv[0] + 1, "loc")) {
            add_localkdb = true;
        } else if (streq(argv[0] + 1, "un")) {
            union_kdb_localkdb = true;
        } else {
            break;
        }

        name_ += argv[0];

        ++argv;
    }
}

dpkdb_recursion_k2::~dpkdb_recursion_k2(void) {
}

void dpkdb_recursion_k2::getCapabilities(capabilities & c) {
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

void dpkdb_recursion_k2::reset(InstanceStream & is) {
    //printf("reset\n");
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
    order_.resize(noCatAtts_);
    order_.clear();
    /*初始化各数据结构空间*/
    dist_.reset(is); //
    dist_1.reset(is); //
    classDist_.reset(is);

    trainingIsFinished_ = false;

    //    arc_sum += arc_sum;
    //    arc_loc_sum += arc_loc_sum;
    //    arc_same_sum += arc_same_sum;
    //    wer += wer;

}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass

/*通过训练集来填写数据空间*/
void dpkdb_recursion_k2::train(const instance & inst) {
    dist_.update(inst);
    dist_1.update(inst);


    classDist_.update(inst);

}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void dpkdb_recursion_k2::initialisePass() {
    assert(trainingIsFinished_ == false);
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void dpkdb_recursion_k2::finalisePass() {
    //printf("finalisePass\n");
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

        for (int a = noCatAtts_ - 1; a >= 0; a--) {
            order_.push_back(order[a]);
        }


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
        order.clear();
        for (unsigned int i = 0; i < noCatAtts_; i++) {
            if (parents_[i].size() == 1) {
                arc++;
            }
            if (parents_[i].size() > 1) {
                arc += k_;
            }
        }

        //实验测试
        //kdb的总弧数
        printf("finalpass begin:\n");
        //printf("%d\t", arc);

    }

    trainingIsFinished_ = true;
}

/// true iff no more passes are required. updated by finalisePass()

bool dpkdb_recursion_k2::trainingIsFinished() {
    return trainingIsFinished_;
}

void dpkdb_recursion_k2::classify(const instance& inst, std::vector<double> &posteriorDist) {
    //sub优化
    std::vector<bool> generalizationSet;
    generalizationSet.assign(noCatAtts_, false);
    //compute the generalisation set and substitution set for
    //lazy subsumption resolution
    if (subsumptionResolution == true) {
        for (CategoricalAttribute i = 1; i < noCatAtts_; i++) {
            const CatValue iVal = inst.getCatVal(i);
            const InstanceCount countOfxi = dist_1.xxyCounts.xyCounts.getCount(i, iVal);

            for (CategoricalAttribute j = 0; j < i; j++) {
                if (!generalizationSet[j]) {
                    const CatValue jVal = inst.getCatVal(j);
                    const InstanceCount countOfxixj = dist_1.xxyCounts.getCount(i, iVal,
                            j, jVal);
                    const InstanceCount countOfxj = dist_1.xxyCounts.xyCounts.getCount(
                            j, jVal);

                    if (countOfxj == countOfxixj && countOfxj >= minCount) {
                        //xi is a generalisation or substitution of xj
                        //once one xj has been found for xi, stop for rest j
                        //generalizationSet[i] = true;
                        fathercount[j]++;
                        break;
                    } else if (countOfxi == countOfxixj
                            && countOfxi >= minCount) {
                        fathercount[i]++;
                        generalizationSet[j] = true;
                    }
                }
            }
        }
    }

    //删加优化操作输入的声明
    parents_0.resize(noCatAtts_);
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        parents_0[a].clear();
    }
    //局部结构的声明
    parents_1.resize(noCatAtts_);
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        parents_1[a].clear();
    }
    //构造局部
    if (add_localkdb == true) {
        std::vector<float> mi_loc;
        getMutualInformationloc(dist_1.xxyCounts.xyCounts, mi_loc, inst);

        // calculate the conditional mutual information from the xxy distribution
        crosstab<float> cmi_loc = crosstab<float>(noCatAtts_);
        getCondMutualInfloc(dist_1.xxyCounts, cmi_loc, inst);

        // sort the attributes on MI with the class
        std::vector<CategoricalAttribute> order1;
        order_loc.resize(noCatAtts_);
        order_loc.clear();
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            order1.push_back(a);
        }

        // assign the parents
        if (!order1.empty()) {
            miCmpClass cmp(&mi_loc);

            std::sort(order1.begin(), order1.end(), cmp);
            order_loc.clear();
            for (int a = noCatAtts_ - 1; a >= 0; a--) {
                order_loc.push_back(order1[a]);
            }

            // proper KDB assignment of parents
            for (std::vector<CategoricalAttribute>::const_iterator it = order1.begin() + 1; it != order1.end(); it++) {
                parents_1[*it].push_back(order1[0]);
                for (std::vector<CategoricalAttribute>::const_iterator it2 = order1.begin() + 1; it2 != it; it2++) {
                    // make parents into the top k attributes on mi that precede *it in order1
                    if (parents_1[*it].size() < k_) {
                        parents_1[*it].push_back(*it2);
                    }
                    for (unsigned int i = 0; i < parents_1[*it].size(); i++) {
                        if (cmi_loc[*it2][*it] > cmi_loc[parents_1[*it][i]][*it]) {
                            for (unsigned int j = parents_1[*it].size() - 1; j > i; j--) {
                                parents_1[*it][j] = parents_1[*it][j - 1];
                            }
                            parents_1[*it][i] = *it2;
                            break;
                        }
                    }
                }
            }
        }

        //parents_0放局部parents_1结构
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            for (unsigned int b = 0; b < parents_1[a].size(); b++) {
                if (!generalizationSet[a]) {
                    parents_0[a].push_back(parents_1[a][b]);
                }
            }
        }

    }
    //parents 为全局结构
    //parents_1为局部结构
    //parents_2为局部结构删加优化后的结构

    if (add_localkdb == false) {
        //parents_0放全局parents_结构
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            for (unsigned int b = 0; b < parents_[a].size(); b++) {
                if (!generalizationSet[a]) {
                    parents_0[a].push_back(parents_[a][b]);
                }
            }
        }
    }
    //对parents_0进行删加优化操作
    double H_standard = 0.0;
    H_standard = H_standard_loc_k2(classDist_, dist_1, parents_0, inst);
    //printf("%lf\n", H_standard);

    parents_2.resize(noCatAtts_);
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        parents_2[a].clear();
    }
    if (add_localkdb == true) {
        H_recursion_dp_k2(classDist_, dist_1, H_standard, arc, order_loc, parents_0, parents_2, inst);
    } else {
        H_recursion_dp_k2(classDist_, dist_1, H_standard, arc, order_, parents_0, parents_2, inst);
    }
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        parents_0[a].clear();
    }

    //实验测试
    int arc_loc = 0; //localkdb的总弧数
    for (unsigned int i = 0; i < noCatAtts_; i++) {
        if (parents_2[i].size() == 1) {
            arc_loc++;
        }
        if (parents_2[i].size() > 1) {
            arc_loc += k_;
        }
    }
    //localkdb与kdb的相同弧数
    int same_count = 0;
    for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
        if (parents_[i].size() == 0 && parents_2[i].size() == 0) {
            same_count++;
        }
        if (parents_[i].size() == 1) {
            for (unsigned int j = 0; j < parents_2[i].size(); j++) {
                if (parents_2[i][j] == parents_[i][0])
                    same_count++;
            }
        }
        if (parents_[i].size() == 2) {
            for (unsigned int i2 = 0; i2 < 2; i2++) {
                for (unsigned int j = 0; j < parents_2[i].size(); j++) {
                    if (parents_2[i][j] == parents_[i][i2])
                        same_count++;
                }
            }
        }
    }
    arc_sum += arc;
    arc_loc_sum += arc_loc;
    arc_same_sum += same_count;
    wer++;
    arc_ratio += 1 - float(arc_same_sum) / arc_loc_sum;
    printf("%llf\t%llf\t%llf\n", arc_sum / wer, arc_loc_sum / wer, arc_ratio / wer);
    //printf("%llf\t%llf\t%llf\t%llf\n", arc_sum, arc_loc_sum, arc_same_sum, arc_ratio);
    //联合概率
    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        for (CatValue y = 0; y < noClasses_; y++) {
            if (parents_2[x1].size() == 0) {
                // printf("PARent=0  \n");
                posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate
            } else if (parents_2[x1].size() == 1) {
                //  printf("PARent=1  \n");
                const InstanceCount totalCount1 = dist_1.xxyCounts.xyCounts.getCount(parents_2[x1][0], inst.getCatVal(parents_2[x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_2[x1][0], inst.getCatVal(parents_2[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents_2[x1].size() == 2) {
                // printf("PARent=2  \n");
                const InstanceCount totalCount1 = dist_1.xxyCounts.getCount(parents_2[x1][0], inst.getCatVal(parents_2[x1][0]), parents_2[x1][1], inst.getCatVal(parents_2[x1][1]));
                if (totalCount1 == 0) {
                    const InstanceCount totalCount2 = dist_1.xxyCounts.xyCounts.getCount(parents_2[x1][0], inst.getCatVal(parents_2[x1][0]));
                    if (totalCount2 == 0) {
                        posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_2[x1][0], inst.getCatVal(parents_2[x1][0]), y);
                    }
                } else {
                    posteriorDist[y] *= dist_1.p(x1, inst.getCatVal(x1), parents_2[x1][0], inst.getCatVal(parents_2[x1][0]), parents_2[x1][1], inst.getCatVal(parents_2[x1][1]), y);
                }

            }
        }
    }
    // normalise the results
    normalise(posteriorDist);


    if (union_kdb_localkdb == true) {
        //全局parents_的联合概率
        std::vector<double> posteriorDist1;
        posteriorDist1.assign(noClasses_, 0);

        //联合概率
        for (CatValue y = 0; y < noClasses_; y++) {
            posteriorDist1[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);
        }

        for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
            for (CatValue y = 0; y < noClasses_; y++) {
                if (parents_[x1].size() == 0) {
                    // printf("PARent=0  \n");
                    posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate
                } else if (parents_[x1].size() == 1) {
                    //  printf("PARent=1  \n");
                    const InstanceCount totalCount1 = dist_1.xxyCounts.xyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]));
                    if (totalCount1 == 0) {
                        posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist1[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                    }
                } else if (parents_[x1].size() == 2) {
                    // printf("PARent=2  \n");
                    const InstanceCount totalCount1 = dist_1.xxyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]), parents_[x1][1], inst.getCatVal(parents_[x1][1]));
                    if (totalCount1 == 0) {
                        const InstanceCount totalCount2 = dist_1.xxyCounts.xyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]));
                        if (totalCount2 == 0) {
                            posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                        } else {
                            posteriorDist1[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), y);
                        }
                    } else {
                        posteriorDist1[y] *= dist_1.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), parents_[x1][1], inst.getCatVal(parents_[x1][1]), y);
                    }

                }
            }
        }
        normalise(posteriorDist1);
        //实验测试，条件概率差异度
        const CatValue prediction_loc = indexOfMaxVal(posteriorDist);
        const CatValue prediction_kdb = indexOfMaxVal(posteriorDist1);
        printf("%d\t%d\t", prediction_loc, prediction_kdb);
//        if (prediction_loc == prediction_kdb)
//            differ += fabs(posteriorDist[prediction_loc] - posteriorDist1[prediction_kdb]);
        if (prediction_loc != prediction_kdb)
            differ += fabs(posteriorDist[prediction_loc] + posteriorDist1[prediction_kdb]);

        printf("%llf\n", differ / wer);

        //联合概率结合
        for (int classno = 0; classno < noClasses_; classno++) {
            posteriorDist[classno] += posteriorDist1[classno];
            posteriorDist[classno] = posteriorDist[classno] / 2;
        }
    }

}