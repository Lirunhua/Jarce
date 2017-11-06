/* 
 * File:   kdb_PH_norder.cpp
 * Author: Administrator
 * 
 * Created on 2016年9月26日, 上午9:57
 */

#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>

#include "kdb_PH_norder.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

kdb_PH_norder::kdb_PH_norder() : pass_(1) {
}

kdb_PH_norder::kdb_PH_norder(char*const*& argv, char*const* end) : pass_(1) {
    name_ = "kdb_PH_norder";

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

kdb_PH_norder::~kdb_PH_norder(void) {
}

void kdb_PH_norder::getCapabilities(capabilities &c) {
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

void kdb_PH_norder::reset(InstanceStream &is) {
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();

    k_ = min(k_, noCatAtts_ - 1); // k cannot exceed the real number of categorical attributes - 1
    kdb_root = 0;
    // initialise distributions
    //dTree_.resize(noCatAtts);
    parents_.resize(noCatAtts);
    for (CategoricalAttribute a = 0; a < noCatAtts; a++) {
        parents_[a].clear();
    }

    order_mi.resize(noCatAtts);
    order_mi.clear();

    /*初始化各数据结构空间*/
    dist_.reset(is); //
    dist_1.reset(is); //
    classDist_.reset(is);
    trainingIsFinished_ = false;
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass

/*通过训练集来填写数据空间*/
void kdb_PH_norder::train(const instance &inst) {
    dist_.update(inst);
    dist_1.update(inst);
    classDist_.update(inst);
}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void kdb_PH_norder::initialisePass() {
    assert(trainingIsFinished_ == false);
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void kdb_PH_norder::finalisePass() {
    assert(trainingIsFinished_ == false);
    // calculate the mutual information from the xy distribution
    std::vector<float> mi;
    getMutualInformation(dist_.xyCounts, mi);

    // calculate the conditional mutual information from the xxy distribution
    crosstab<float> cmi = crosstab<float>(noCatAtts_);
    getCondMutualInf(dist_, cmi);

    dist_.clear();

    // sort the attributes on MI with the class
    std::vector<CategoricalAttribute> order;

    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        order.push_back(a);
    }

    // assign the parents
    if (!order.empty()) {
        miCmpClass cmp(&mi);

        std::sort(order.begin(), order.end(), cmp);

        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            //printf("%d\t", order[a]);
            order_mi.push_back(order[a]);
        }//printf("\n");
        kdb_root = order[0];
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
    order.clear();
    trainingIsFinished_ = true;
}

/// true iff no more passes are required. updated by finalisePass()

bool kdb_PH_norder::trainingIsFinished() {
    return trainingIsFinished_;
}

int countT(std::vector<bool> sign) {
    int count = 0;
    for (CategoricalAttribute a = 0; a < sign.size(); a++) {
        if (sign[a] == true)
            count++;
    }
    return count;
}

void kdb_PH_norder::classify(const instance& inst, std::vector<double> &posteriorDist) {
    parents_temp.resize(noCatAtts_);
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        parents_temp[a].clear();
    }
    printf("kdb:\n");
    printf("P(x0,...,xn)\tH(Y)-H(Y|x1,...,xn)\tP(x0,...,xn)*{H(Y)-H(Y|x1,...,xn)}\n");
    for (std::vector<CategoricalAttribute>::const_iterator it = order_mi.begin() + 1; it != order_mi.end(); it++) {
        //printf("%d\t",*it);
        if (parents_[*it].size() == 1)
            parents_temp[*it].push_back(parents_[*it][0]);
        else if (parents_[*it].size() == 2) {
            parents_temp[*it].push_back(parents_[*it][0]);
            parents_temp[*it].push_back(parents_[*it][1]);
        }
        //double Ho = H_PH_only_have_parents(classDist_, dist_1, kdb_root, parents_temp, inst);
        double Ho = H_PH(classDist_, dist_1, parents_temp, inst);
        printf("%lle\n", Ho);
    }
    printf("\n");

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

    hmiCmpClass hcmp(&hmi);
    std::sort(horder.begin(), horder.end(), hcmp);
    //    printf("horder:\n");
    //    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
    //        printf("%d\t", horder[a]);
    //    }
    //    printf("\n");

    std::vector<CategoricalAttribute> temp;
    temp.clear();
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        temp.push_back(a);
    }

    std::vector<bool> sign;
    sign.assign(noCatAtts_, false);
    CategoricalAttribute root = horder[0];

    sign[root] = true;
   
    //double H_max = 0.0;
    double H_max = H_PH(classDist_, dist_1, parents_1, inst);
    printf("\nroot=%d\tH_max=%lle\n",root,H_max);
    CategoricalAttribute pos0 = 0xFFFFFFFFUL;
    for (std::vector<CategoricalAttribute>::const_iterator it = temp.begin(); it != temp.end(); it++) {
        //遍历没确定加入结构的属性
        if (!sign[*it]) {
            //printf("%d\t", *it);
            parents_1[*it].push_back(root);
            //double Ho = H_PH_only_have_parents(classDist_, dist_1, root, parents_1, inst);
            double Ho = H_PH(classDist_, dist_1, parents_1, inst);
            for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
                parents_1[a].clear();
            }
            //printf("``Ho=%lle\tpos1=%d\n", Ho, *it);
            if (Ho > H_max) {
                H_max = Ho;
                pos0 = *it;
            }
        }
    }
    printf("\n");
    printf("pos0=%d\tH_max=%lle\n", pos0, H_max);
    parents_2[pos0].push_back(root);
    sign[pos0] = true;

    int count = countT(sign); //countT计算已确定加入结构的属性的个数
    while (count != noCatAtts_) {

        for (std::vector<CategoricalAttribute>::const_iterator it = temp.begin(); it != temp.end(); it++) {
            printf("it=%d    count=%d\n",*it, count);
            if (!sign[*it]) {
                CategoricalAttribute pos1 = 0xFFFFFFFFUL;
                CategoricalAttribute pos2 = 0xFFFFFFFFUL;
                for (std::vector<CategoricalAttribute>::const_iterator it1 = temp.begin(); it1 != temp.end(); it1++) {
                    for (std::vector<CategoricalAttribute>::const_iterator it2 = it1 + 1; it2 != temp.end(); it2++) {
                        //printf("%d:%d+%d\n", *it, *it1, *it2);
                        //遍历没确定加入结构的属性
                        if (sign[*it1] && sign[*it2]) {
                            //printf("\n%d:%d+%d\n", *it, *it1, *it2);
                            //printf("%d\t", *it);
                            for (unsigned int i = 0; i < noCatAtts_; i++) {
                                for (unsigned int j = 0; j < parents_2[i].size(); j++) {
                                    parents_1[i].push_back(parents_2[i][j]);
                                }
                            }
                            parents_1[*it].push_back(*it1);
                            parents_1[*it].push_back(*it2);
                            //double Ho = H_PH_only_have_parents(classDist_, dist_1, root, parents_1, inst);
                            double Ho = H_PH(classDist_, dist_1, parents_1, inst);
                            for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
                                parents_1[a].clear();
                            }
                            //printf("``Ho=%lle\tpos1=%d\n", Ho, *it);
                            if (Ho > H_max) {
                                H_max = Ho;
                                pos1 = *it1;
                                pos2 = *it2;
                            }
                        }
                    }
                }
                if (pos1 != 0xFFFFFFFFUL && pos2 != 0xFFFFFFFFUL) {
                    printf("p=%d+%d\tH_max=%lle\n", pos1, pos2, H_max);
                    parents_2[*it].push_back(pos1);
                    parents_2[*it].push_back(pos2);
                    sign[*it] = true;
                } else {
                    sign[*it] = true;
                }
                count = countT(sign);
            }
        }
    }

        printf("parents_2:\n");
        for (unsigned int i = 0; i < noCatAtts_; i++) {
            if (parents_2[i].size() == 0) {
                printf("parents2[%d][0]=Y\n", i);
            }
            if (parents_2[i].size() == 1) {
                printf("parents2[%d][0]=%d\n", i, parents_2[i][0]);
            }
            if (parents_2[i].size() == 2) {
                printf("parents2[%d][0]=%d\tparents2[%d][1]=%d\n", i, parents_2[i][0], i, parents_2[i][1]);
            }
        }
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        parents_temp[a].clear();
    }
    printf("\nlocalkdb:\n");
    printf("P(x0,...,xn)\tH(Y)-H(Y|x1,...,xn)\tP(x0,...,xn)*{H(Y)-H(Y|x1,...,xn)}\n");
    for (std::vector<CategoricalAttribute>::const_iterator it = order_mi.begin() + 1; it != order_mi.end(); it++) {
        //printf("%d\t",*it);
        if (parents_2[*it].size() == 1)
            parents_temp[*it].push_back(parents_2[*it][0]);
        else if (parents_2[*it].size() == 2) {
            parents_temp[*it].push_back(parents_2[*it][0]);
            parents_temp[*it].push_back(parents_2[*it][1]);
        }
        //double Ho = H_PH_only_have_parents(classDist_, dist_1, root, parents_temp, inst);
        double Ho = H_PH(classDist_, dist_1, parents_temp, inst);
        printf("%lle\n", Ho);
    }
    printf("\n");
    printf("real:%d\n", inst.getClass());

    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist_.p(y);
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
    double m = 0.0;
    for (CatValue y = 0; y < noClasses_; y++) {
        m += posteriorDist[y];
    }
    double H_standard = 0.0;
    for (CatValue y = 0; y < noClasses_; y++) {
        H_standard -= (posteriorDist[y] / m) * log2(posteriorDist[y] / m);
        printf("`p[%d]=%lf\t",y,posteriorDist[y] / m);

    }printf("\n");
    printf("m=%lle\tH=%lle\n", m, H_standard);
    normalise(posteriorDist);
    printf("locEnt_kdb:\t");
    for (CatValue y = 0; y < noClasses_; y++) {
        printf("p[%d]=%lf\t", y, posteriorDist[y]);
    }
    printf("\n");
    //    //**********
    std::vector<double> posteriorDist1;
    posteriorDist1.assign(noClasses_, 0);


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
    printf("kdb:\t\t");
    for (CatValue y = 0; y < noClasses_; y++) {
        printf("p[%d]=%lf\t", y, posteriorDist1[y]);
    }
    printf("\n");
    //**********
    double newp = (posteriorDist[0] + 1) / (posteriorDist[1] + 1);
    double kdbp = (posteriorDist1[0] + 1) / (posteriorDist1[1] + 1);
    if (newp < 1 && kdbp < 1) {
        printf("same\n");
    } else if (newp > 1 && kdbp > 1) {
        printf("same\n");
    } else {
        printf("different\n");
    }
    printf("------------------------------------------\n");

    // normalise the results
    //normalise(posteriorDist);


}








