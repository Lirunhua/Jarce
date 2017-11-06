/* 
 * File:   sortkdb.cpp
 * Author: Administrator
 * 
 * Created on 2016年12月3日, 下午9:43
 */
#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>

#include "sortkdb.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

sortkdb::sortkdb() : pass_(1) {
}

sortkdb::sortkdb(char*const*& argv, char*const* end) : pass_(1) {
    name_ = "sortkdb";

    // defaults
    k_ = 1;

    sum_unmi = 0.0;
    sum_uncmi = 0.0;
    union_localkdb = false;


    // get arguments
    while (argv != end) {
        if (*argv[0] != '-') {
            break;
        } else if (argv[0][1] == 'k') {
            getUIntFromStr(argv[0] + 2, k_, "k");
        } else if (streq(argv[0] + 1, "unloc")) {
            union_localkdb = true;
        } else {
            break;
        }

        name_ += argv[0];

        ++argv;
    }
}

sortkdb::~sortkdb(void) {
}

void sortkdb::getCapabilities(capabilities &c) {
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

class miCmpClass11 {
public:

    miCmpClass11(std::vector<float> *m) {
        mi = m;
    }

    bool operator()(CategoricalAttribute a, CategoricalAttribute b) {
        return (*mi)[a] > (*mi)[b];
    }

private:
    std::vector<float> *mi;
};

void sortkdb::reset(InstanceStream &is) {
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
void sortkdb::train(const instance &inst) {
    //    printf("train:\n");
    dist_.update(inst);
    dist_1.update(inst);
    classDist_.update(inst);
}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void sortkdb::initialisePass() {
    assert(trainingIsFinished_ == false);
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void sortkdb::finalisePass() {
    //printf("--------------------finalPass:------------------------\n");
    assert(trainingIsFinished_ == false);
    // calculate the mutual information from the xy distribution
    std::vector<float> mi; //互信息
    getMutualInformation(dist_.xxyCounts.xyCounts, mi);

    //联合互信息，二维
    crosstab<float> mi_xxy = crosstab<float>(noCatAtts_); //联合互信息I(Xi,Xj;Y)
    getUnionMI(dist_.xxyCounts, mi_xxy);

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
                //printf("%lf\t%d\n",sum,j);
                if (maxunmi < sum) {
                    maxunmi = sum;
                    node = j;
                }
            }
        }
        //printf("-------->%lf\t%d\n",maxunmi,node);
        order.push_back(node);
        InOrder[node] = true;
    }
    //            printf("order:\t");
    //            for (CategoricalAttribute a = 0; a < order.size(); a++) {
    //                printf("%d\t", order[a]);
    //            }
    //            printf("\n");

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
                //                                printf("%d\t:\t%d\t%d\t", ch, fa1, fa2);
                //                                printf("%lf\n", un_cmi[ch][fa1][fa2]);
                parents_[ch].push_back(fa1);
                parents_[ch].push_back(fa2);
                used_nodes.push_back(ch);
                waiting_nodes[ch] = false;
            }
        }
    }
    //        printf("parents_incpp:\n");
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

bool sortkdb::trainingIsFinished() {
    return trainingIsFinished_;
}

void sortkdb::classify(const instance& inst, std::vector<double> &posteriorDist) {
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
    normalise(posteriorDist);
    //局部
    if (union_localkdb == true) {
        parents_1.resize(noCatAtts_);
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            parents_1[a].clear();
        }
        std::vector<float> mi_loc; //互信息
        getMutualInformationloc(dist_1.xxyCounts.xyCounts, mi_loc, inst);

        //联合互信息，二维
        crosstab<float> mi_xxy_loc = crosstab<float>(noCatAtts_); //局部联合互信息I(xi,xj;Y)
        getUnmi_loc(dist_1.xxyCounts, mi_xxy_loc, inst);   

        crosstab3D<float> un_cmi_loc = crosstab3D<float>(noCatAtts_); //局部联合条件互信息I(xi,xj;xk|Y)
        getUnionCmi_loc(dist_1, un_cmi_loc, inst);    

        // sort the attributes on MI with the class
        std::vector<CategoricalAttribute> order_loc;
        std::vector<bool> InOrder_loc;
        InOrder_loc.assign(noCatAtts_, false);

        order_loc.clear();
        double maxmi = -std::numeric_limits<double>::max();
        CategoricalAttribute root = 0xFFFFFFFFUL;
        for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
            if (maxmi < mi_loc[i]) {
                maxmi = mi_loc[i];
                root = i;
            }
        }
        order_loc.push_back(root);
        InOrder_loc[root] = true;

        double maxunmi = -std::numeric_limits<double>::max();
        CategoricalAttribute root1 = 0xFFFFFFFFUL;
        for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
            if (i != order_loc[0]) {
                if (maxunmi < mi_xxy_loc[order_loc[0]][i]) {
                    maxunmi = mi_xxy_loc[order_loc[0]][i];
                    root1 = i;
                }
            }
        }
        order_loc.push_back(root1);
        InOrder_loc[root1] = true;

        unsigned int boundary_loc = noCatAtts_;

        while (order_loc.size() != boundary_loc) {
            maxunmi = -std::numeric_limits<double>::max();
            CategoricalAttribute node = 0xFFFFFFFFUL;
            for (CategoricalAttribute j = 0; j < noCatAtts_; j++) {
                double sum = 0.0;
                if (!InOrder_loc[j]) {
                    for (CategoricalAttribute i = 0; i < order_loc.size(); i++) {
                        sum += mi_xxy_loc[order_loc[i]][j];
                    }
                    if (maxunmi < sum) {
                        maxunmi = sum;
                        node = j;
                    }
                }
            }
            order_loc.push_back(node);
            InOrder_loc[node] = true;
        }

        std::vector<CategoricalAttribute> used_nodes_loc;
        std::vector<bool> waiting_nodes_loc;
        waiting_nodes_loc.assign(noCatAtts_, true);

        used_nodes_loc.push_back(order_loc[0]);
        used_nodes_loc.push_back(order_loc[1]);
        parents_1[order_loc[1]].push_back(order_loc[0]);
        waiting_nodes_loc[order_loc[0]] = false;
        waiting_nodes_loc[order_loc[1]] = false;
        // proper KDB assignment of parents
        while (used_nodes_loc.size() != boundary_loc) {
            CategoricalAttribute fa1 = 0xFFFFFFFFUL;
            CategoricalAttribute fa2 = 0xFFFFFFFFUL;
            CategoricalAttribute ch = 0xFFFFFFFFUL;
            for (std::vector<CategoricalAttribute>::const_iterator it = order_loc.begin(); it != order_loc.end(); it++) {
                double maxuncmi = -std::numeric_limits<double>::max();
                if (waiting_nodes_loc[*it]) {
                    for (CategoricalAttribute xi = 0; xi < used_nodes_loc.size(); xi++) { //待选父节点
                        for (CategoricalAttribute xj = xi + 1; xj < used_nodes_loc.size(); xj++) {
                            if (maxuncmi < un_cmi_loc[*it][used_nodes_loc[xi]][used_nodes_loc[xj]]) {
                                maxuncmi = un_cmi_loc[*it][used_nodes_loc[xi]][used_nodes_loc[xj]];
                                fa1 = used_nodes_loc[xi];
                                fa2 = used_nodes_loc[xj];
                                ch = *it;
                            }
                        }
                    }
                    parents_1[ch].push_back(fa1);
                    parents_1[ch].push_back(fa2);
                    used_nodes_loc.push_back(ch);
                    waiting_nodes_loc[ch] = false;
                }
            }
        }



        std::vector<double> posteriorDist1;
        posteriorDist1.assign(noClasses_, 0);

        //联合概率
        for (CatValue y = 0; y < noClasses_; y++) {
            posteriorDist1[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);
        }
        for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
            for (CatValue y = 0; y < noClasses_; y++) {
                if (parents_1[x1].size() == 0) {
                    posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate
                } else if (parents_1[x1].size() == 1) {
                    const InstanceCount totalCount1 = dist_1.xxyCounts.xyCounts.getCount(parents_1[x1][0], inst.getCatVal(parents_1[x1][0]));
                    if (totalCount1 == 0) {
                        posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist1[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_1[x1][0], inst.getCatVal(parents_1[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                    }
                } else if (parents_1[x1].size() == 2) {
                    const InstanceCount totalCount1 = dist_1.xxyCounts.getCount(parents_1[x1][0], inst.getCatVal(parents_1[x1][0]), parents_1[x1][1], inst.getCatVal(parents_1[x1][1]));
                    if (totalCount1 == 0) {
                        const InstanceCount totalCount2 = dist_1.xxyCounts.xyCounts.getCount(parents_1[x1][0], inst.getCatVal(parents_1[x1][0]));
                        if (totalCount2 == 0) {
                            posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                        } else {
                            posteriorDist1[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_1[x1][0], inst.getCatVal(parents_1[x1][0]), y);
                        }
                    } else {
                        posteriorDist1[y] *= dist_1.p(x1, inst.getCatVal(x1), parents_1[x1][0], inst.getCatVal(parents_1[x1][0]), parents_1[x1][1], inst.getCatVal(parents_1[x1][1]), y);
                    }
                }
            }
        }
        // normalise the results
        normalise(posteriorDist1);

        //联合概率结合
        for (int classno = 0; classno < noClasses_; classno++) {
            posteriorDist[classno] += posteriorDist1[classno];
            posteriorDist[classno] = posteriorDist[classno] / 2;
        }
    }

}

void sortkdb::getNoCatAtts_(unsigned int &NoCatAtt) {
    //printf(" getNoCatAtts_ \n");
    NoCatAtt = noCatAtts_;
}

void sortkdb::getStructure(std::vector<std::vector<CategoricalAttribute> > &parents, std::vector<CategoricalAttribute> &order) {
    for (unsigned int i = 0; i < noCatAtts_; i++) {
        for (unsigned int j = 0; j < parents_[i].size(); j++) {
            parents[i].push_back(parents_[i][j]);
        }
    }

    for (unsigned int i = 0; i < noCatAtts_; i++) {
        order.push_back(order_[i]);
    }

}

void sortkdb::classify_change(const instance& inst, std::vector<double> &posteriorDist, std::vector<std::vector<CategoricalAttribute> > &parents_) {
    // calculate the class probabilities in parallel
    // P(y)

    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);

    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        for (CatValue y = 0; y < noClasses_; y++) {
            if (parents_[x1].size() == 0) {
                posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate
            } else if (parents_[x1].size() == 1) {
                const InstanceCount totalCount1 = dist_1.xxyCounts.xyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents_[x1].size() == 2) {
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
    //printf("have changed\n");
}

void sortkdb::chang_parents(std::vector<std::vector<CategoricalAttribute> > &parents_change) {
    for (unsigned int i = 0; i < noCatAtts_; i++)
        parents_[i].clear();
    for (unsigned int i = 0; i < noCatAtts_; i++) {
        for (unsigned int j = 0; j < parents_change[i].size(); j++) {
            parents_[i].push_back(parents_change[i][j]);
        }
    }
}





