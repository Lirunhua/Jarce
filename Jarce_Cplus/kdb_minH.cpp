/* 
 * File:   kdb_minH.cpp
 * Author: Administrator
 * 
 * Created on 2016年10月16日, 上午10:42
 */
#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>

#include "kdb_minH.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

kdb_minH::kdb_minH() : pass_(1) {
}

kdb_minH::kdb_minH(char*const*& argv, char*const* end) : pass_(1) {
    name_ = "kdb_minH";

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

kdb_minH::~kdb_minH(void) {
}

void kdb_minH::getCapabilities(capabilities &c) {
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

void kdb_minH::reset(InstanceStream &is) {
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
    }
    /*初始化各数据结构空间*/
    dist_.reset(is); //
    dist_1.reset(is); //
    classDist_.reset(is);
    trainingIsFinished_ = false;
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass

/*通过训练集来填写数据空间*/
void kdb_minH::train(const instance &inst) {
    dist_.update(inst);
    dist_1.update(inst);
    classDist_.update(inst);
}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void kdb_minH::initialisePass() {
    assert(trainingIsFinished_ == false);
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void kdb_minH::finalisePass() {
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

bool kdb_minH::trainingIsFinished() {
    return trainingIsFinished_;
}

void kdb_minH::classify(const instance& inst, std::vector<double> &posteriorDist) {
    // printf("classify\n");
    parents_1.resize(noCatAtts_); //最终返回的结构
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        parents_1[a].clear();
    }
    parents_2.resize(noCatAtts_); //过渡结构
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        parents_2[a].clear();
    }

    double H0 = H_standard_loc_k2(classDist_, dist_1, parents_1, inst);
    double H_nb = H0;
    printf("\t\t\tH_nb=%lf\t", H0);
    printf("-\t");
    displayInfo(inst, parents_1);

    //相关变量初始化
    std::vector<CategoricalAttribute> readyBeParentsOrder; //待选父节点
    readyBeParentsOrder.clear();
    std::vector<bool> sign;
    sign.assign(noCatAtts_, false);

    double H_temp = 0.0;
    CategoricalAttribute pos_A = 0xFFFFFFFFUL; //A->B
    CategoricalAttribute pos_B = 0xFFFFFFFFUL;
    //找出第一条使得熵下降的弧
    for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
        for (CategoricalAttribute j = i + 1; j < noCatAtts_; j++) {
            //printf("%d\t%d\n",i,j);
            parents_1[j].push_back(i); //i->j,j的父结点为i
            H_temp = H_standard_loc_k2(classDist_, dist_1, parents_1, inst);
            if (H_temp < H0) {
                pos_A = i;
                pos_B = j;
                H0 = H_temp;
            }
            for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
                parents_1[a].clear();
            }
        }
    }
    parents_1[pos_B].push_back(pos_A);
    readyBeParentsOrder.push_back(pos_A);
    readyBeParentsOrder.push_back(pos_B);
    sign[pos_A] = true;
    sign[pos_B] = true;

    printf("plus: %d->%d\t\tH0=%lf\t", pos_A, pos_B, H0);
    if (H0 > H_nb) {
        printf("↗\t"); //↗↘
    } else
        printf("↘\t");
    displayInfo(inst, parents_1);

    while (readyBeParentsOrder.size() != noCatAtts_) {
        //printf("readyBeParentsOrder.size=%d\n", readyBeParentsOrder.size());

        for (unsigned int a = 0; a < noCatAtts_; a++) {
            for (unsigned int b = 0; b < parents_1[a].size(); b++) {
                parents_2[a].push_back(parents_1[a][b]);
            }
        }
        CategoricalAttribute pos_A1 = readyBeParentsOrder[0]; //A1->B
        CategoricalAttribute pos_A2 = readyBeParentsOrder[1]; //A2->B
        CategoricalAttribute x = 0;
        while (sign[x])
            x++;
        pos_B = x;

        parents_2[pos_B].push_back(pos_A1);
        parents_2[pos_B].push_back(pos_A2);
        double H_AB = H_standard_loc_k2(classDist_, dist_1, parents_2, inst);
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            parents_2[a].clear();
        }
        for (unsigned int a = 0; a < noCatAtts_; a++) {
            for (unsigned int b = 0; b < parents_1[a].size(); b++) {
                parents_2[a].push_back(parents_1[a][b]);
            }
        }
        //printf("suppose plus : %d->%d\t%d->%d\tH_temp0=%lf\n", pos_A1, pos_B, pos_A2, pos_B, H_AB);

        H_temp = 0.0;
        for (CategoricalAttribute i = 0; i < noCatAtts_; i++) { //j->i,j为父,一次加两条弧
            if (!sign[i]) {
                for (CategoricalAttribute j1 = 0; j1 < readyBeParentsOrder.size(); j1++) {
                    for (CategoricalAttribute j2 = j1 + 1; j2 < readyBeParentsOrder.size(); j2++) {
                        //printf("%d\t%d\t%d\n", i, j1, j2);
                        parents_2[i].push_back(readyBeParentsOrder[j1]);
                        parents_2[i].push_back(readyBeParentsOrder[j2]);
                        H_temp = H_standard_loc_k2(classDist_, dist_1, parents_2, inst);
                        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
                            parents_2[a].clear();
                        }
                        for (unsigned int a = 0; a < noCatAtts_; a++) {
                            for (unsigned int b = 0; b < parents_1[a].size(); b++) {
                                parents_2[a].push_back(parents_1[a][b]);
                            }
                        }
                        //printf("suppose plus : %d->%d\t%d->%d\tH_temp=%lf\n", readyBeParentsOrder[j1], i, readyBeParentsOrder[j2], i, H_temp);
                        if (H_temp < H_AB) {
                            pos_A1 = readyBeParentsOrder[j1];
                            pos_A2 = readyBeParentsOrder[j2];
                            pos_B = i;
                            H_AB = H_temp;
                        }

                    }
                }
            }
        }
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            parents_2[a].clear();
        }
        parents_1[pos_B].push_back(pos_A1);
        parents_1[pos_B].push_back(pos_A2);

        readyBeParentsOrder.push_back(pos_B);
        sign[pos_B] = true;

        printf("plus: %d->%d\t%d->%d\tH=%lf\t", pos_A1, pos_B, pos_A2, pos_B, H_AB);

        if (H_AB > H0) {
            printf("↗\t"); //↗↘
        } else
            printf("↘\t");
        H0 = H_AB;
        displayInfo(inst, parents_1);
    }

    printf("readyBeParentsOrder.size=%d is full! over\n", readyBeParentsOrder.size());
    printf("----------\n");
    printf("real:%d\n", inst.getClass());

    printf("kdb:\t\t");
    displayInfo(inst, parents_);
    printf("localkdb:\t");
    displayInfo(inst, parents_1);



    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = dist_1.xxyCounts.xyCounts.p(y)* (std::numeric_limits<double>::max() / 2.0);

    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        for (CatValue y = 0; y < noClasses_; y++) {
            if (parents_1[x1].size() == 0) {

                posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate

            } else if (parents_1[x1].size() == 1) {
                const InstanceCount totalCount1 = dist_1.xxyCounts.xyCounts.getCount(parents_1[x1][0], inst.getCatVal(parents_1[x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_1[x1][0], inst.getCatVal(parents_1[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents_1[x1].size() == 2) {
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
    //    printf("localkdb:\t");
    //    double maxposteriorDist = 0.0;
    //    CatValue pre = 0;
    //    for (CatValue y = 0; y < noClasses_; y++) {
    //        if(maxposteriorDist < posteriorDist[y]){
    //            maxposteriorDist = posteriorDist[y];
    //            pre = y;
    //        }
    //    }
    //    printf("%d\t",pre);
    //    for (CatValue y = 0; y < noClasses_; y++) {
    //        printf("p[%d]=%lf\t", y, posteriorDist[y]);
    //    }
    //    printf("\n");
    //    
    //    //**********************
    //    std::vector<double> posteriorDist1;
    //    posteriorDist1.assign(noClasses_, 0);
    //
    //
    //    for (CatValue y = 0; y < noClasses_; y++) {
    //        posteriorDist1[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);
    //
    //    }
    //    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
    //        for (CatValue y = 0; y < noClasses_; y++) {
    //            if (parents_[x1].size() == 0) {
    //                // printf("PARent=0  \n");
    //
    //                posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate
    //
    //
    //            } else if (parents_[x1].size() == 1) {
    //                //  printf("PARent=1  \n");
    //                const InstanceCount totalCount1 = dist_1.xxyCounts.xyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]));
    //                if (totalCount1 == 0) {
    //                    posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
    //                } else {
    //                    posteriorDist1[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
    //                }
    //            } else if (parents_[x1].size() == 2) {
    //                // printf("PARent=2  \n");
    //                const InstanceCount totalCount1 = dist_1.xxyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]), parents_[x1][1], inst.getCatVal(parents_[x1][1]));
    //                if (totalCount1 == 0) {
    //                    const InstanceCount totalCount2 = dist_1.xxyCounts.xyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]));
    //                    if (totalCount2 == 0) {
    //                        posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
    //                    } else {
    //                        posteriorDist1[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), y);
    //                    }
    //                } else {
    //                    posteriorDist1[y] *= dist_1.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), parents_[x1][1], inst.getCatVal(parents_[x1][1]), y);
    //                }
    //
    //            }
    //        }
    //    }
    //    normalise(posteriorDist1);
    //    printf("kdb:\t\t");
    //    double maxposteriorDist1 = 0.0;
    //    CatValue pre1 = 0;
    //    for (CatValue y = 0; y < noClasses_; y++) {
    //        if(maxposteriorDist1 < posteriorDist1[y]){
    //            maxposteriorDist1 = posteriorDist1[y];
    //            pre1 = y;
    //        }
    //    }
    //    printf("%d\t",pre1);
    //    for (CatValue y = 0; y < noClasses_; y++) {
    //        printf("p[%d]=%lf\t", y, posteriorDist1[y]);
    //    }
    //    printf("\n");
    //    //**********
    //    if (pre != pre1)
    //        printf("different\n");
    //    else
    //        printf("same\n");
    //    double newp = (posteriorDist[0] + 1) / (posteriorDist[1] + 1);
    //    double kdbp = (posteriorDist1[0] + 1) / (posteriorDist1[1] + 1);
    //    if (newp < 1 && kdbp < 1) {
    //        printf("same\n");
    //    } else if (newp > 1 && kdbp > 1) {
    //        printf("same\n");
    //    } else {
    //        printf("different\n");
    //    }
    printf("------------------------------------------\n");
}

void kdb_minH::displayInfo(const instance& inst, std::vector<std::vector<CategoricalAttribute> > parents_) {
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

    double H_standard = 0.0;
    double m = 0.0;
    for (CatValue y = 0; y < noClasses_; y++) {
        m += posteriorDist1[y];
    }
    for (CatValue y = 0; y < noClasses_; y++) {
        H_standard -= (posteriorDist1[y] / m) * log2(posteriorDist1[y] / m);
    }
    //printf("%lf\t",H_standard);
    normalise(posteriorDist1);
    double maxposteriorDist1 = 0.0;
    CatValue pre1 = 0xFFFFFFFFUL;
    for (CatValue y = 0; y < noClasses_; y++) {
        if (posteriorDist1[y] > maxposteriorDist1) {
            maxposteriorDist1 = posteriorDist1[y];
            pre1 = y;
        }
    }
    printf("%d\t", pre1);
    for (CatValue y = 0; y < noClasses_; y++) {
        printf("p[%d]=%lf\t", y, posteriorDist1[y]);
    }
    printf("\n");
}







