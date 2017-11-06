/* 
 * File:   dpkdbrec.cpp
 * Author: Administrator
 * 
 * Created on 2016年11月29日, 下午2:54
 */
#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>

#include "dpkdbrec.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

dpkdbrec::dpkdbrec() : pass_(1) {
}

dpkdbrec::dpkdbrec(char*const*& argv, char*const* end) : pass_(1) {
    name_ = "dpkdbrec";

    subsumptionResolution = false;
    add_localkdb = false;
    union_kdb_localkdb = false;
    minCount = 100;

    //printf("!!\n");
    arc_sum = 0; //kdb弧数
    arc_loc_sum = 0; //localkdb弧数
    arc_same_sum = 0; //二者相同弧段数目
    wer = 0; //分类总次数,测试实例总数
    differ = 0.0; //条件概率差值
    posteriorDist_diff = 0.0; //条件概率差异度 = 条件概率差值/kdb与localkdb分类结果不同数，/wer，取平均
    arc_ratio = 0.0; //结构差异度，（1-相同弧数/local弧数）/wer，取平均

    hunhe_correct_num = 0; //混合分类正确数
    hunhe_wrong_num = 0; //混合分类错误数
    diff_num = 0; //kdb与localkdb分类结果不同数
    kdb_correct_num = 0; //kdb分类正确数
    kdb_wrong_num = 0; //kdb分类错误数
    hunhe_correct_num_bydiff = 0; //混合分类正确数,在kdb与localkdb分类结果不同情况下

    hubulv = 0.0; //互补率 = 混合对/kdb与localkdb分类结果不同数
    wupanlv = 0.0; //误判率 = 在kdb分类正确的情况下混合分类错误数 / wer
    jiucuolv = 0.0; //纠错率 = 在kdb分类错误的情况下混合分类正确数 / wer


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

dpkdbrec::~dpkdbrec(void) {
}

void dpkdbrec::getCapabilities(capabilities & c) {
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

// creates a comparator for two attributes based on their relative mutual information with the class

class miCmpClassdp {
public:

    miCmpClassdp(std::vector<float> *m) {
        mi = m;
    }

    bool operator()(CategoricalAttribute a, CategoricalAttribute b) {
        return (*mi)[a] > (*mi)[b]; //mi
    }

private:
    std::vector<float> *mi;
};

void dpkdbrec::reset(InstanceStream & is) {
    //printf("reset\n");
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();
    arc = 0;
    k_ = min(k_, noCatAtts_ - 1); // k cannot exceed the real number of categorical attributes - 1

    // initialise distributions
    parents_.resize(noCatAtts);
    for (CategoricalAttribute a = 0; a < noCatAtts; a++) {
        parents_[a].clear();
    }
    order_.resize(noCatAtts_);
    order_.clear();
    /*初始化各数据结构空间*/
    dist_.reset(is); //
    dist_1.reset(is); //
    classDist_.reset(is);

    trainingIsFinished_ = false;
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass

/*通过训练集来填写数据空间*/
void dpkdbrec::train(const instance & inst) {
    dist_.update(inst);
    dist_1.update(inst);
    classDist_.update(inst);
}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void dpkdbrec::initialisePass() {
    assert(trainingIsFinished_ == false);
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void dpkdbrec::finalisePass() {
    assert(trainingIsFinished_ == false);
    // calculate the mutual information from the xy distribution
    std::vector<float> mi;
    getMutualInformation(dist_.xxyCounts.xyCounts, mi);

    // calculate the conditional mutual information from the xxy distribution
    crosstab<float> cmi = crosstab<float>(noCatAtts_);
    getCondMutualInf(dist_.xxyCounts, cmi);

    dist_.clear();

    // sort the attributes on MI with the class
    std::vector<CategoricalAttribute> order;
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        order.push_back(a);
    }

    // assign the parents
    if (!order.empty()) {
        miCmpClassdp cmp(&mi);

        std::sort(order.begin(), order.end(), cmp);

        for (int a = noCatAtts_ - 1; a >= 0; a--) {
            order_.push_back(order[a]);
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


    }

    trainingIsFinished_ = true;
}

/// true iff no more passes are required. updated by finalisePass()

bool dpkdbrec::trainingIsFinished() {
    return trainingIsFinished_;
}

void dpkdbrec::classify(const instance& inst, std::vector<double> &posteriorDist) {
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
                posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate
            } else if (parents_[x1].size() == 1) {
                const InstanceCount totalCount1 = dist_1.xxyCounts.xyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist1[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents_[x1].size() == 2) {
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


    //构造局部
    if (add_localkdb == true) {
        //局部结构的声明
        parents_1.resize(noCatAtts_);
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            parents_1[a].clear();
        }
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
            miCmpClassdp cmp(&mi_loc);

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
        //对parents_0进行删加优化操作
        double H_standard = 0.0;
        H_standard = H_standard_loc_k2(classDist_, dist_1, parents_1, inst);

        parents_2.resize(noCatAtts_);
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            parents_2[a].clear();
        }
        H_recursion_dp_k2(classDist_, dist_1, H_standard, arc, order_loc, parents_1, parents_2, inst);

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


        //联合概率
        for (CatValue y = 0; y < noClasses_; y++) {
            posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);
        }
        for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
            for (CatValue y = 0; y < noClasses_; y++) {
                if (parents_2[x1].size() == 0) {
                    posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate
                } else if (parents_2[x1].size() == 1) {
                    const InstanceCount totalCount1 = dist_1.xxyCounts.xyCounts.getCount(parents_2[x1][0], inst.getCatVal(parents_2[x1][0]));
                    if (totalCount1 == 0) {
                        posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_2[x1][0], inst.getCatVal(parents_2[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                    }
                } else if (parents_2[x1].size() == 2) {
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
    }


    if (union_kdb_localkdb == true) {
        //实验测试，条件概率差异度
        const CatValue prediction_loc = indexOfMaxVal(posteriorDist);
        const CatValue prediction_kdb = indexOfMaxVal(posteriorDist1);
        //printf("%d\t%d\t", prediction_loc, prediction_kdb);
        //        if (prediction_loc == prediction_kdb)
        //            differ += fabs(posteriorDist[prediction_loc] - posteriorDist1[prediction_kdb]);
        if (prediction_loc != prediction_kdb) {
            differ += fabs(posteriorDist[prediction_loc] + posteriorDist1[prediction_kdb]);
            diff_num++; //kdb与localkdb分类结果不同数
        }
        if (diff_num != 0)
            posteriorDist_diff += differ / diff_num;
        printf("condition_diff = %llf\n", posteriorDist_diff / wer / 2);

        if (prediction_kdb == inst.getClass())
            kdb_correct_num++; //kdb分类正确数
        else
            kdb_wrong_num++; //kdb分类错误数

        //联合概率结合
        for (int classno = 0; classno < noClasses_; classno++) {
            posteriorDist[classno] += posteriorDist1[classno];
            posteriorDist[classno] = posteriorDist[classno] / 2;
        }

        const CatValue prediction_hunhe = indexOfMaxVal(posteriorDist);
        //实验测试，互补率
        if (prediction_loc != prediction_kdb && prediction_hunhe == inst.getClass()) {
            hunhe_correct_num_bydiff++;
        }
        if (diff_num != 0)
            hubulv += float(hunhe_correct_num_bydiff) / diff_num;
        printf("hubulv = %llf\n", hubulv / wer);
        //实验测试，误判率
        //误判率 = 在kdb分类正确的情况下混合分类错误数 / wer
        if (prediction_kdb == inst.getClass() && prediction_hunhe != inst.getClass())
            hunhe_wrong_num++;
        if (kdb_correct_num != 0)
            wupanlv += float(hunhe_wrong_num) / kdb_correct_num;
        printf("wupanlv = %llf\n", wupanlv / wer);

        //实验测试，纠错率
        //纠错率 = 在kdb分类错误的情况下混合分类正确数 / wer
        if (prediction_kdb != inst.getClass() && prediction_hunhe == inst.getClass())
            hunhe_correct_num++;
        if (kdb_wrong_num != 0)
            jiucuolv += float(hunhe_correct_num) / kdb_wrong_num;
        printf("jiucuolv = %llf\n", jiucuolv / wer);
    }
    if (union_kdb_localkdb == false && add_localkdb == false) {
        for (int classno = 0; classno < noClasses_; classno++) {
            posteriorDist[classno] = posteriorDist1[classno];
        }
    }

}
























