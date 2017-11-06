/* 
 * File:   sortkdb_del.cpp
 * Author: Administrator
 * 
 * Created on 2016年12月14日, 上午10:18
 */
#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>

#include "sortkdb_del.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

sortkdb_del::sortkdb_del() : pass_(1) {
}

sortkdb_del::sortkdb_del(char*const*& argv, char*const* end) : pass_(1) {
    name_ = "sortkdb_del";

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

sortkdb_del::~sortkdb_del(void) {
}

void sortkdb_del::getCapabilities(capabilities &c) {
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

// creates a comparator for two attributes based on their relative mutual information with the class

class miCmpClass1 {
public:

    miCmpClass1(std::vector<double> *m) {
        mi = m;
    }

    bool operator()(CategoricalAttribute a, CategoricalAttribute b) {
        return (*mi)[a] < (*mi)[b]; //小到大
    }
private:
    std::vector<double> *mi;
};

void sortkdb_del::reset(InstanceStream &is) {
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();

    k_ = min(k_, noCatAtts_ - 1); // k cannot exceed the real number of categorical attributes - 1

    // initialise distributions
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
void sortkdb_del::train(const instance &inst) {
    //    printf("train:\n");
    dist_.update(inst);
    dist_1.update(inst);
    classDist_.update(inst);
}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void sortkdb_del::initialisePass() {
    assert(trainingIsFinished_ == false);
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void sortkdb_del::finalisePass() {
    //printf("--------------------finalPass:------------------------\n");
    assert(trainingIsFinished_ == false);
    // calculate the mutual information from the xy distribution
    std::vector<float> mi; //互信息
    getMutualInformation(dist_.xxyCounts.xyCounts, mi);
    //    printf("mi:\t");
    //        std::vector<CategoricalAttribute> order1;
    //        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
    //            order1.push_back(a);
    //        }
    //        miCmpClass cmp(&mi);
    //        std::sort(order1.begin(), order1.end(), cmp);
    //        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
    //            printf("%d\t",order1[a]);
    //        }
    //        printf("\n");


    // calculate the conditional mutual information from the xxy distribution
    crosstab<float> cmi = crosstab<float>(noCatAtts_); //条件互信息
    getCondMutualInf(dist_.xxyCounts, cmi);

    //联合互信息，二维
    crosstab<float> mi_xxy = crosstab<float>(noCatAtts_); //联合互信息I(Xi,Xj;Y)
    getUnionMI(dist_.xxyCounts, mi_xxy);

    crosstab<float> cmi_xyx = crosstab<float>(noCatAtts_); //非类条件互信息I(Xi,Y|Xj)
    crosstab<float> cmi_xyx_ratio = crosstab<float>(noCatAtts_);
    getCMIxyx(dist_.xxyCounts, cmi_xyx, cmi_xyx_ratio);

    crosstab3D<float> un_cmi = crosstab3D<float>(noCatAtts_); //联合条件互信息I(Xi,Xj;Xk|Y)
    getUnionCmi(dist_, un_cmi);
    dist_.clear();
    //    for (CategoricalAttribute k = 0; k < noCatAtts_; k++) {
    //        printf("k=%d\n",k);
    //        for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
    //            for (CategoricalAttribute j = 0; j < noCatAtts_; j++) {
    //                printf("%lf\t",un_cmi[k][i][j]);
    //            }
    //            printf("\n");
    //        }
    //        printf("\n");
    //    }


    // sort the attributes on MI with the class
    std::vector<CategoricalAttribute> order;
    std::vector<bool> InOrder;
    InOrder.assign(noCatAtts_, false);

    order.clear();
    double maxmi = -std::numeric_limits<double>::max();
    CategoricalAttribute root = 0xFFFFFFFFUL;
    for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
        if (maxmi < mi[i]) {
            maxmi = mi[i];
            root = i;
        }
    }
    order.push_back(root);
    InOrder[root] = true;

    double maxunmi = -std::numeric_limits<double>::max();
    CategoricalAttribute root1 = 0xFFFFFFFFUL;
    for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
        if (i != order[0]) {
            if (maxunmi < mi_xxy[order[0]][i]) {
                maxunmi = mi_xxy[order[0]][i];
                root1 = i;
            }
        }
    }
    order.push_back(root1);
    InOrder[root1] = true;

    unsigned int boundary = noCatAtts_;

    while (order.size() != boundary) {
        maxunmi = -std::numeric_limits<double>::max();
        CategoricalAttribute node = 0xFFFFFFFFUL;
        for (CategoricalAttribute j = 0; j < noCatAtts_; j++) {
            double sum = 0.0;
            if (!InOrder[j]) {
                for (CategoricalAttribute i = 0; i < order.size(); i++) {
                    sum += mi_xxy[order[i]][j];
                }
                if (maxunmi < sum) {
                    maxunmi = sum;
                    node = j;
                }
            }
        }
        order.push_back(node);
        InOrder[node] = true;
    }
    //    printf("order:\t");
    //    for (CategoricalAttribute a = 0; a < order.size(); a++) {
    //        printf("%d\t", order[a]);
    //    }
    //    printf("\n");


    for (int a = noCatAtts_ - 1; a >= 0; a--) {
        order_.push_back(order[a]);
    }

    std::vector<CategoricalAttribute> used_nodes;
    std::vector<bool> waiting_nodes;
    waiting_nodes.assign(noCatAtts_, true);

    used_nodes.push_back(order[0]);
    used_nodes.push_back(order[1]);
    parents_[order[1]].push_back(order[0]);
    //    printf("%d\n", order[0]);
    //    printf("%d\t:\t%d\n", order[1], order[0]);
    waiting_nodes[order[0]] = false;
    waiting_nodes[order[1]] = false;
    // proper KDB assignment of parents
    while (used_nodes.size() != boundary) {
        CategoricalAttribute fa1 = 0xFFFFFFFFUL;
        CategoricalAttribute fa2 = 0xFFFFFFFFUL;
        CategoricalAttribute ch = 0xFFFFFFFFUL;
        for (std::vector<CategoricalAttribute>::const_iterator it = order.begin(); it != order.end(); it++) {
            double maxuncmi = -std::numeric_limits<double>::max();
            if (waiting_nodes[*it]) {
                for (CategoricalAttribute xi = 0; xi < used_nodes.size(); xi++) { //待选父节点
                    for (CategoricalAttribute xj = xi + 1; xj < used_nodes.size(); xj++) {
                        if (maxuncmi < un_cmi[*it][used_nodes[xi]][used_nodes[xj]]) {
                            maxuncmi = un_cmi[*it][used_nodes[xi]][used_nodes[xj]];
                            fa1 = used_nodes[xi];
                            fa2 = used_nodes[xj];
                            ch = *it;
                        }
                    }
                }
                //                printf("%d\t:\t%d\t%d\t", ch, fa1, fa2);
                //                printf("%lf\n", un_cmi[ch][fa1][fa2]);
                parents_[ch].push_back(fa1);
                parents_[ch].push_back(fa2);
                used_nodes.push_back(ch);
                waiting_nodes[ch] = false;
            }
        }
    }

    //    printf("parents_incpp:\n");
    //    for (unsigned int i = 0; i < noCatAtts_; i++) {
    //        if (parents_[i].size() == 0) {
    //            printf("parents_[%d][0]\tY\n", i);
    //        }
    //        if (parents_[i].size() == 1) {
    //            printf("parents_[%d][0]=%d\n", i, parents_[i][0]);
    //        }
    //        if (parents_[i].size() == 2) {
    //            printf("parents_[%d][0]=%d\tparents_[%d][1]=%d\n", i, parents_[i][0], i, parents_[i][1]);
    //        }
    //    }

    std::vector<CategoricalAttribute> delorder;
    for (int a = order.size() - 3; a >= 0; a--) {
        delorder.push_back(order_[a]);
        //printf("%d\t", order_[a]);
    }
    //printf("\n");

    double unmi_xxy_sum = 0.0;
    double uncmi_xxy_sum = 0.0;

    for (std::vector<CategoricalAttribute>::const_iterator it = delorder.begin(); it != delorder.end(); it++) {
        CategoricalAttribute i = *it;
        unmi_xxy_sum += mi_xxy[i][parents_[i][0]] + mi_xxy[i][parents_[i][1]]; //联合互信息I(Xi,Xj;Y)
        uncmi_xxy_sum += un_cmi[i][parents_[i][0]][parents_[i][1]]; //联合条件互信息I(Xi,Xj;Xk|Y)
    }
    //    printf("total unmi:%lf\n", unmi_xxy_sum);
    //    printf("total uncmi:%lf\n", uncmi_xxy_sum);

    std::vector<double> mi_percent;
    mi_percent.resize(noCatAtts_);
    for (unsigned int i = 0; i < noCatAtts_; i++) {
        mi_percent[i] = 10;
    }
    for (std::vector<CategoricalAttribute>::const_iterator it = delorder.begin(); it != delorder.end(); it++) {
        mi_percent[*it] = ((mi_xxy[*it][parents_[*it][0]] + mi_xxy[*it][parents_[*it][1]]) / unmi_xxy_sum);
    }

    std::vector<double> cmi_percent;
    cmi_percent.resize(noCatAtts_);
    for (unsigned int i = 0; i < noCatAtts_; i++) {
        cmi_percent[i] = 10;
    }
    for (std::vector<CategoricalAttribute>::const_iterator it = delorder.begin(); it != delorder.end(); it++) {
        cmi_percent[*it] = (un_cmi[*it][parents_[*it][0]][parents_[*it][1]] / uncmi_xxy_sum);
    }

    std::vector<CategoricalAttribute> order1;
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        order1.push_back(a);
    }
    miCmpClass1 cmp1(&mi_percent);
    std::sort(order1.begin(), order1.end(), cmp1);

    std::vector<CategoricalAttribute> order2;
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        order2.push_back(a);
    }
    miCmpClass1 cmp2(&cmi_percent);
    std::sort(order2.begin(), order2.end(), cmp2);

    std::sort(mi_percent.begin(), mi_percent.end());
    std::sort(cmi_percent.begin(), cmi_percent.end());
    //    printf("order1:\t\t");
    //    for (unsigned int i = 0; i < noCatAtts_; i++)
    //        printf("%d\t", order1[i]);
    //    printf("\n");
    //   printf("unmi_order:\t");
    //    for (unsigned int i = 0; i < noCatAtts_; i++) {
    //        if (mi_percent[i] != 10)
    //            printf("%lf\t", mi_percent[i]);
    //        else
    //            printf("max\t");
    //    }
    //    printf("\n");
    //    printf("\n");
    //
    //    printf("order2:\t\t");
    //    for (unsigned int i = 0; i < noCatAtts_; i++)
    //        printf("%d\t", order2[i]);
    //    printf("\n");
    //    printf("uncmi_order:\t");
    //    for (unsigned int i = 0; i < noCatAtts_; i++) {
    //        if (cmi_percent[i] != 10)
    //            printf("%lf\t", cmi_percent[i]);
    //        else
    //            printf("max\t");
    //    }
    //    printf("\n");

    double threshold = 0.05;
    double temp = 0.0;
    std::vector<CategoricalAttribute> ret1;
    int i = 0;
    while (temp < threshold) {
        temp += mi_percent[i];
        if (temp < threshold)
            ret1.push_back(order1[i]);
        i++;
    }

    double temp2 = 0.0;
    std::vector<CategoricalAttribute> ret2;
    i = 0;
    while (temp2 < threshold) {
        temp2 += cmi_percent[i];
        if (temp2 < threshold)
            ret2.push_back(order2[i]);
        i++;
    }

    std::vector<bool> del;
    del.assign(noCatAtts_, false);
    for (CategoricalAttribute i = 0; i < ret1.size(); i++) {
        for (CategoricalAttribute j = 0; j < ret2.size(); j++) {
            if (ret1[i] == ret2[j])
                del[ret1[i]] = true;
        }
    }

    //    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
    //        if(del[a])
    //            printf("%d\t",a);
    //    }
    //    printf("\n");
    std::vector<std::vector<CategoricalAttribute> > parents_change; //声明改变后的结构
    parents_change.resize(noCatAtts_);
    for (unsigned int i = 0; i < noCatAtts_; i++) {
        if (del[i])
            continue;
        else {
            for (unsigned int j = 0; j < parents_[i].size(); j++) {
                parents_change[i].push_back(parents_[i][j]);
            }
        }
    }
    for (unsigned int i = 0; i < noCatAtts_; i++)
        parents_[i].clear();
    for (unsigned int i = 0; i < noCatAtts_; i++) {
        for (unsigned int j = 0; j < parents_change[i].size(); j++) {
            parents_[i].push_back(parents_change[i][j]);
        }
    }
    for (unsigned int i = 0; i < noCatAtts_; i++)
        parents_change[i].clear();

    delorder.clear();
    mi_percent.clear();
    cmi_percent.clear();
    //printf("--------------------------------------------------------\n");
//         printf("parents_incpp:\n");
//        for (unsigned int i = 0; i < noCatAtts_; i++) {
//            if (parents_[i].size() == 0) {
//                printf("parents_[%d][0]\tY\n", i);
//            }
//            if (parents_[i].size() == 1) {
//                printf("parents_[%d][0]=%d\n", i, parents_[i][0]);
//            }
//            if (parents_[i].size() == 2) {
//                printf("parents_[%d][0]=%d\tparents_[%d][1]=%d\n", i, parents_[i][0], i, parents_[i][1]);
//            }
//        }
    trainingIsFinished_ = true;
}


/// true iff no more passes are required. updated by finalisePass()

bool sortkdb_del::trainingIsFinished() {
    return trainingIsFinished_;
}

void sortkdb_del::classify(const instance& inst, std::vector<double> &posteriorDist) {
    // calculate the class probabilities in parallel
    // P(y)

    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);

    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        for (CatValue y = 0; y < noClasses_; y++) {

            if (parents_[x1].size() == 0) {
                // printf("PARent=0  \n");

                posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate

            } else if (parents_[x1].size() == 1) {
                //  printf("PARent=1  \n");
                const InstanceCount totalCount1 = dist_1.xxyCounts.xyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents_[x1].size() == 2) {
                // printf("PARent=2  \n");
                const InstanceCount totalCount1 = dist_1.xxyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]), parents_[x1][1], inst.getCatVal(parents_[x1][1]));
                if (totalCount1 == 0) {
                    const InstanceCount totalCount2 = dist_1.xxyCounts.xyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]));
                    if (totalCount2 == 0) {
                        posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), y);
                    }
                } else {
                    posteriorDist[y] *= dist_1.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), parents_[x1][1], inst.getCatVal(parents_[x1][1]), y);
                }

            }
        }
    }


    // normalise the results
    normalise(posteriorDist);
}




