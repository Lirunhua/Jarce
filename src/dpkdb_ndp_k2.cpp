/* 
 * File:   dpkdb_ndp_k2.cpp
 * Author: Skyrim
 * 
 * Created on 2016年8月30日, 下午12:29
 */
#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>

#include "dpkdb_ndp_k2.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

dpkdb_ndp_k2::dpkdb_ndp_k2() : pass_(1) {
}

dpkdb_ndp_k2::dpkdb_ndp_k2(char*const*& argv, char*const* end) : pass_(1) {
    name_ = "dpkdb_ndp_k2";

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

dpkdb_ndp_k2::~dpkdb_ndp_k2(void) {
}

void dpkdb_ndp_k2::getCapabilities(capabilities &c) {
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

void dpkdb_ndp_k2::reset(InstanceStream &is) {
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
    //pass_ = 1;
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass

/*通过训练集来填写数据空间*/
void dpkdb_ndp_k2::train(const instance &inst) {
    dist_.update(inst);
    dist_1.update(inst);


    classDist_.update(inst);

}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void dpkdb_ndp_k2::initialisePass() {
    assert(trainingIsFinished_ == false);
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void dpkdb_ndp_k2::finalisePass() {
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
        //        printf("mi:\t");
        //        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        //            printf("%d\t", order[a]);
        //        }
        //        printf("\n");
        //        printf("order_:\t");
        for (int a = noCatAtts_ - 1; a >= 0; a--) {
            //printf("%d\t", order[a]);
            order_.push_back(order[a]);
        }
        //printf("\n");
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
                arc += 2;
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
        //        printf("\n");

    }

    trainingIsFinished_ = true;
}

/// true iff no more passes are required. updated by finalisePass()

bool dpkdb_ndp_k2::trainingIsFinished() {
    return trainingIsFinished_;
}

void dpkdb_ndp_k2::classify(const instance& inst, std::vector<double> &posteriorDist) {
    parents_0.resize(noCatAtts_);
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        parents_0[a].clear();
    }
    for (unsigned int i = 0; i < noCatAtts_; i++) {
        for (unsigned int j = 0; j < parents_[i].size(); j++) {
            parents_0[i].push_back(parents_[i][j]);
        }
    }

    double H_standard = 0.0;
    H_standard = H_standard_loc_k2(classDist_, dist_1, parents_0, inst);
    printf("H=%lf\t", H_standard);
    displayInfo(inst, parents_0);
    //    printf("----------------\n");
    //    printf("H_1=%lf\n", H_standard);
    int loop = 0;
    while (loop < 3) {
        //printf("H_1(%d)=%lf\n", loop, H_standard);
        //printf("\n");
        //printf("H_1=%lf\n", H_standard);
        parents_1.resize(noCatAtts_);
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            parents_1[a].clear();
        }

        unsigned int k = 0;
        double gm = H_standard;
        //unsigned int pos = 0;

        std::vector<bool> pos;
        pos.resize(arc);
        pos.assign(arc, false);
        while (k < arc) {
            int temp = 0;
            for (unsigned int i = 0; i < noCatAtts_; i++) {
                for (unsigned int j = 0; j < parents_0[i].size(); j++) {
                    //printf("temp=%d\n", temp);
                    if (temp == k) {
                        temp++;
                        continue;
                    } else {
                        temp++;
                        parents_1[i].push_back(parents_0[i][j]);
                    }
                }
            }
            //            printf("parents_1:\n");
            //            for (unsigned int i = 0; i < noCatAtts_; i++) {
            //                if (parents_1[i].size() == 1) {
            //                    printf("parents1[%d][0]=%d\n", i, parents_1[i][0]);
            //                }
            //                if (parents_1[i].size() > 1) {
            //                    printf("parents1[%d][0]=%d\tparents1[%d][1]=%d\n", i, parents_1[i][0], i, parents_1[i][1]);
            //                }
            //            }
            //            printf("\n");

            double H_new = 0.0;
            H_new = H_standard_loc_k2(classDist_, dist_1, parents_1, inst);
            if (H_new < gm) {
                //gm = H_new;
                pos[k] = true;
            }

            for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
                parents_1[a].clear();
            }
            k++;
        }
        int m = 0;
        for (unsigned int i = 0; i < noCatAtts_; i++) {
            for (unsigned int j = 0; j < parents_0[i].size(); j++) {
                //printf("temp=%d\n", temp);
                if (pos[m]) {
                    m++;
                    continue;
                } else {
                    m++;
                    parents_1[i].push_back(parents_0[i][j]);
                }
            }
        }
        pos.clear();

        //        printf("parents_1   del:\n");
        //        for (unsigned int i = 0; i < noCatAtts_; i++) {
        //            if (parents_1[i].size() == 1) {
        //                printf("parents1[%d][0]=%d\n", i, parents_1[i][0]);
        //            }
        //            if (parents_1[i].size() > 1) {
        //                printf("parents1[%d][0]=%d\tparents1[%d][1]=%d\n", i, parents_1[i][0], i, parents_1[i][1]);
        //            }
        //        }
        //printf("\n");
        double H_2 = 0.0; //全删后的局部条件熵
        H_2 = H_standard_loc_k2(classDist_, dist_1, parents_1, inst);
        //printf("H_2=%lf\n", H_2);
        //printf("\n");

        //加弧************************************************************************
        parents_2.resize(noCatAtts_);
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            parents_2[a].clear();
        }

        parents_3.resize(noCatAtts_);
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            parents_3[a].clear();
        }

        for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); it != order_.end(); it++) {
            double gm_plus = H_2;
            double gm_plus2 = H_2;
            int direct = 2;
            bool finded = false;
            bool finded1 = false;
            CategoricalAttribute posl = 0xFFFFFFFFUL;
            CategoricalAttribute posl1 = 0xFFFFFFFFUL;
            CategoricalAttribute posl2 = 0xFFFFFFFFUL;
            for (std::vector<CategoricalAttribute>::const_iterator it2 = it + 1; it2 != order_.end(); it2++) {
                for (unsigned int i = 0; i < noCatAtts_; i++) {
                    for (unsigned int j = 0; j < parents_1[i].size(); j++) {
                        parents_2[i].push_back(parents_1[i][j]);
                    }
                }
                //printf("%d  %d\n", *it, *it2);

                if (parents_1[*it].size() == 2) {
                    //printf("have two arc\n");
                } else if (parents_1[*it].size() == 1) {
                    direct = 1;
                    //printf("have one arc\n");
                    if (parents_2[*it][0] != *it2) {
                        parents_2[*it].push_back(*it2);
                        double H_3 = 0.0; //加一条弧后的局部条件熵
                        H_3 = H_standard_loc_k2(classDist_, dist_1, parents_2, inst);
                        if (H_3 < gm_plus) {
                            //printf("%d  %d 's H < H_standard\n", *it, *it2);
                            gm_plus = H_3;
                            posl = *it2;
                            finded = true;
                        }
                    }
                } else if (parents_1[*it].size() == 0) {
                    direct = 0;
                    //printf("have zero arc\n");
                    parents_2[*it].push_back(*it2);
                    double H_3_1 = 0.0; //加一条弧后的局部条件熵
                    H_3_1 = H_standard_loc_k2(classDist_, dist_1, parents_2, inst);
                    if (H_3_1 < gm_plus2) {
                        //printf("%d  %d 's H < H_standard\n", *it, *it2);
                        gm_plus2 = H_3_1;
                        posl2 = posl1;
                        posl1 = *it2;
                        finded1 = true;
                    }

                }
                for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
                    parents_2[a].clear();
                }
            }
            if (direct == 1 && finded == true && posl != 0xFFFFFFFFUL) {
                //printf("1   parents_3[%d][1].pushback(%d)\n", *it, posl);
                parents_3[*it].push_back(posl);
            }
            if (direct == 0 && finded1 == true && posl1 != 0xFFFFFFFFUL) {
                //printf("0   parents_3[%d][0].pushback(%d)\t", *it, posl1);
                parents_3[*it].push_back(posl1);
                if (direct == 0 && posl2 != 0xFFFFFFFFUL) {
                    //printf("0   parents_3[%d][1].pushback(%d)", *it, posl2);
                    parents_3[*it].push_back(posl2);
                }
            }
            //printf("\n");
        }
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            parents_0[a].clear();
        }


        for (unsigned int i = 0; i < noCatAtts_; i++) {
            for (unsigned int j = 0; j < parents_3[i].size(); j++) {
                parents_1[i].push_back(parents_3[i][j]);
            }
        }
        for (unsigned int i = 0; i < noCatAtts_; i++) {
            for (unsigned int j = 0; j < parents_1[i].size(); j++) {
                parents_0[i].push_back(parents_1[i][j]);
            }
        }
        double H_3 = 0.0;
        H_3 = H_standard_loc_k2(classDist_, dist_1, parents_0, inst);
        //        printf("parents_plus:\n");
        //        for (unsigned int i = 0; i < noCatAtts_; i++) {
        //            if (parents_0[i].size() == 1) {
        //                printf("parents0[%d][0]=%d\n", i, parents_0[i][0]);
        //            }
        //            if (parents_0[i].size() > 1) {
        //                printf("parents0[%d][0]=%d\tparents0[%d][1]=%d\n", i, parents_0[i][0], i, parents_0[i][1]);
        //            }
        //        }
        H_standard = H_3;
        //printf("H_3=%lf\n", H_3);
        loop++;
    }
    //    printf("H_1=%lf\n", H_standard);
    //    printf("H_2=%lf\n", H_2);
    //    printf("H_3=%lf\n", H_3);
    //    printf("H_1(%d)=%lf\n", loop, H_standard);
    //printf("----------------------------------\n");
    //    printf("\n");
    //    printf("parents_final:\n");
    //    for (unsigned int i = 0; i < noCatAtts_; i++) {
    //        if (parents_0[i].size() == 1) {
    //            printf("parents0[%d][0]=%d\n", i, parents_0[i][0]);
    //        }
    //        if (parents_0[i].size() > 1) {
    //            printf("parents0[%d][0]=%d\tparents0[%d][1]=%d\n", i, parents_0[i][0], i, parents_0[i][1]);
    //        }
    //    }


    //    for (unsigned int i = 0; i < noCatAtts_; i++) {
    //        for (unsigned int j = 0; j < parents_1[i].size(); j++) {
    //            if (depthParent(dist_1, parents_0[i][j], i, parents_0)) {
    //                printf("exist loop\n");
    //                break;
    //            }
    //        }
    //    }
    //printf("###########################\n");


    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        for (CatValue y = 0; y < noClasses_; y++) {
            if (parents_0[x1].size() == 0) {
                // printf("PARent=0  \n");

                posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate


            } else if (parents_0[x1].size() == 1) {
                //  printf("PARent=1  \n");
                const InstanceCount totalCount1 = dist_1.xxyCounts.xyCounts.getCount(parents_0[x1][0], inst.getCatVal(parents_0[x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_0[x1][0], inst.getCatVal(parents_0[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents_0[x1].size() == 2) {
                // printf("PARent=2  \n");
                const InstanceCount totalCount1 = dist_1.xxyCounts.getCount(parents_0[x1][0], inst.getCatVal(parents_0[x1][0]), parents_0[x1][1], inst.getCatVal(parents_0[x1][1]));
                if (totalCount1 == 0) {
                    const InstanceCount totalCount2 = dist_1.xxyCounts.xyCounts.getCount(parents_0[x1][0], inst.getCatVal(parents_0[x1][0]));
                    if (totalCount2 == 0) {
                        posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_0[x1][0], inst.getCatVal(parents_0[x1][0]), y);
                    }
                } else {
                    posteriorDist[y] *= dist_1.p(x1, inst.getCatVal(x1), parents_0[x1][0], inst.getCatVal(parents_0[x1][0]), parents_0[x1][1], inst.getCatVal(parents_0[x1][1]), y);
                }

            }
        }
    }
    // normalise the results
    normalise(posteriorDist);
    
    printf("----------\n");
    printf("real:%d\n", inst.getClass());
    printf("localkdb:\t");
    double maxposteriorDist = 0.0;
    CatValue pre = 0;
    for (CatValue y = 0; y < noClasses_; y++) {
        if (maxposteriorDist < posteriorDist[y]) {
            maxposteriorDist = posteriorDist[y];
            pre = y;
        }
    }
    printf("%d\t", pre);
    for (CatValue y = 0; y < noClasses_; y++) {
        printf("p[%d]=%lf\t", y, posteriorDist[y]);
    }
    printf("\n");

    //**********************
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
    double maxposteriorDist1 = 0.0;
    CatValue pre1 = 0;
    for (CatValue y = 0; y < noClasses_; y++) {
        if (maxposteriorDist1 < posteriorDist1[y]) {
            maxposteriorDist1 = posteriorDist1[y];
            pre1 = y;
        }
    }
    printf("%d\t", pre1);
    for (CatValue y = 0; y < noClasses_; y++) {
        printf("p[%d]=%lf\t", y, posteriorDist1[y]);
    }
    printf("\n");
    //**********
    if (pre != pre1)
        printf("different\n");
    else
        printf("same\n");
    
    printf("------------------------------------------\n");
}


void dpkdb_ndp_k2::displayInfo(const instance& inst, std::vector<std::vector<CategoricalAttribute> > parents_) {
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
