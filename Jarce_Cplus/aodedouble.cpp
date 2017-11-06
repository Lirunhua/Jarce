/* Petal: An open source system for classification learning from very large data
 ** Copyright (C) 2012 Geoffrey I Webb
 **
 ** This program is free software: you can redistribute it and/or modify
 ** it under the terms of the GNU General Public License as published by
 ** the Free Software Foundation, either version 3 of the License, or
 ** (at your option) any later version.
 **
 ** This program is distributed in the hope that it will be useful,
 ** but WITHOUT ANY WARRANTY; without even the implied warranty of
 ** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 ** GNU General Public License for more details.
 **
 ** You should have received a copy of the GNU General Public License
 ** along with this program. If not, see <http://www.gnu.org/licenses/>.
 **
 ** Please report any bugs to Shenglei Chen <tristan_chen@126.com>
 */

#include "aodeselect.h"
#include <assert.h>
#include "utils.h"
#include <algorithm>
#include "correlationMeasures.h"
#include "globals.h"
#include "utils.h"
#include <set>
#include "aodedouble.h"

aodeDouble::aodeDouble(char* const *& argv, char* const * end) {
    name_ = "aodedouble";
    UsedAttrRatio = 0;
    weighted = false;
    minCount = 100;
    subsumptionResolution = false;
    selected = false;
    su_ = false;
    mi_ = false;
    chisq_ = false;
    empiricalMEst_ = false;
    empiricalMEst2_ = false;
    for (int i = 0; i < 100; i++) {
        fathercount[i] = 0;
    }
    threshold = 0.8;
    factor_ = 2;


    // get arguments
    while (argv != end) {
        if (*argv[0] != '-') {
            break;
        } else if (streq(argv[0] + 1, "empirical")) {
            empiricalMEst_ = true;
        } else if (streq(argv[0] + 1, "empirical2")) {
            empiricalMEst2_ = true;
        } else if (streq(argv[0] + 1, "sub")) {
            subsumptionResolution = true;
        } else if (argv[0][1] == 'n') {
            getUIntFromStr(argv[0] + 2, minCount, "n");
        } else if (streq(argv[0] + 1, "w")) {
            weighted = true;
        } else if (streq(argv[0] + 1, "selective")) {
            selected = true;
        } else if (streq(argv[0] + 1, "mi")) {
            selected = true;
            mi_ = true;
        } else if (argv[0][1] == 'f') {
            getUIntFromStr(argv[0] + 2, factor_, "f");
        } else if (streq(argv[0] + 1, "su")) {
            selected = true;
            su_ = true;
        } else if (streq(argv[0] + 1, "chisq")) {
            selected = true;
            chisq_ = true;
        } else {
            error("Aode does not support argument %s\n", argv[0]);
            break;
        }

        name_ += *argv;

        ++argv;
    }
    if (selected == true) {
        if (mi_ == false && su_ == false && chisq_ == false)
            chisq_ = true;
    }

    trainingIsFinished_ = false;
}

aodeDouble::~aodeDouble(void) {
}

void aodeDouble::getCapabilities(capabilities &c) {
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void aodeDouble::reset(InstanceStream &is) {
    xxxyDist_.reset(is);
    trainingIsFinished_ = false;
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();
    inactiveCnt_ = 0;

    noCatAtts_ = is.getNoCatAtts();
    noClasses_ = is.getNoClasses();

    weight.assign(noCatAtts_, 1);
    //selectedAtt.resize(noCatAtts_, true);

    instanceStream_ = &is;
    parents_.resize(noCatAtts);
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        parents_[a] = NOPARENT;
    }
    effect_children.resize(is.getNoCatAtts());

    for (CategoricalAttribute x1 = 0; x1 < is.getNoCatAtts(); x1++) {
        effect_children[x1].resize(is.getNoCatAtts());
    }

    for (int i = 0; i < noCatAtts_; i++) {
        for (int j = 0; j < noCatAtts_; j++) {
            effect_children[i][j] = false;
        }
    }

    xxxyDist_.reset(is);
    active_.assign(noCatAtts_, false);

}

void aodeDouble::initialisePass() {

}

void aodeDouble::train(const instance &inst) {
    xxxyDist_.update(inst);
}

/// true iff no more passes are required. updated by finalisePass()

bool aodeDouble::trainingIsFinished() {
    return trainingIsFinished_;
}

void aodeDouble::classify(const instance &inst, std::vector<double> &classDist) {

    std::vector<bool> generalizationSet;    

    generalizationSet.assign(noCatAtts_, false);

    //compute the generalisation set and substitution set for
    //lazy subsumption resolution
    if (subsumptionResolution == true) {
        //********************************************************************************************            
        for (CategoricalAttribute i = 1; i < noCatAtts_ - 1; i++) {

            const CatValue iVal = inst.getCatVal(i);
            for (CategoricalAttribute j = 0; j < i; j++) {
                for (CategoricalAttribute k = i + 1; k < noCatAtts_; k++) {
                    const CatValue jVal = inst.getCatVal(j);
                    const CatValue kVal = inst.getCatVal(k);
                    const InstanceCount countOfxixjxk = xxxyDist_.getCount(i, iVal, j, jVal, k, kVal);
                    const InstanceCount countOfxixj = xxxyDist_.xxyCounts.getCount(i, iVal, j, jVal);
                    const InstanceCount countOfxixk = xxxyDist_.xxyCounts.getCount(i, iVal, k, kVal);
                    const InstanceCount countOfxjxk = xxxyDist_.xxyCounts.getCount(j, jVal, k, kVal);
                    const InstanceCount countOfxj = xxxyDist_.xxyCounts.xyCounts.getCount(j, jVal);
                    const InstanceCount countOfxk = xxxyDist_.xxyCounts.xyCounts.getCount(k, kVal);
                    if (countOfxj == countOfxixj && countOfxj >= minCount) {
                        generalizationSet[i] = true;
                    } else if (countOfxk == countOfxixk && countOfxj != countOfxixj && countOfxk >= minCount) {
                        generalizationSet[i] = true;
                    } else if (countOfxjxk == countOfxixjxk && countOfxjxk >= minCount) {
                        generalizationSet[i] = true;
                    }
                }
            }

        }

//        const CatValue headVal = inst.getCatVal(0);
//        for (CategoricalAttribute j = 1; j < noCatAtts_ - 1; j++) {
//            //	if (!generalizationSet[j]) {
//            for (CategoricalAttribute k = j + 1; k < noCatAtts_; k++) {
//                const CatValue jVal = inst.getCatVal(j);
//                const CatValue kVal = inst.getCatVal(k);
//                const InstanceCount countOfxixjxk = xxxyDist_.getCount(0, headVal, j, jVal, k, kVal);
//                const InstanceCount countOfxixj = xxxyDist_.xxyCounts.getCount(0, headVal, j, jVal);
//                const InstanceCount countOfxixk = xxxyDist_.xxyCounts.getCount(0, headVal, k, kVal);
//                const InstanceCount countOfxjxk = xxxyDist_.xxyCounts.getCount(j, jVal, k, kVal);
//                const InstanceCount countOfxj = xxxyDist_.xxyCounts.xyCounts.getCount(j, jVal);
//                const InstanceCount countOfxk = xxxyDist_.xxyCounts.xyCounts.getCount(k, kVal);
//                if (countOfxjxk == countOfxixjxk && countOfxjxk >= minCount) {
//                    //xi is a generalisation or substitution of xj
//                    //once one xj has been found for xi, stop for rest j
//                    generalizationSet[0] = true;
//                } else if (countOfxj == countOfxixj && countOfxj >= minCount) {
//                    //	xj is a generalisation of xi
//                    generalizationSet[0] = true;
//                } else if (countOfxk == countOfxixk && countOfxk >= minCount) {
//                    //xj is a generalisation of xi
//                    generalizationSet[0] = true;
//                }
//            }
//        }
//
//        const CatValue tailVal = inst.getCatVal(noCatAtts_ - 1);
//        for (CategoricalAttribute j = 0; j < noCatAtts_ - 2; j++) {
//            //	if (!generalizationSet[j]) {
//            for (CategoricalAttribute k = j + 1; k < noCatAtts_ - 1; k++) {
//                const CatValue jVal = inst.getCatVal(j);
//                const CatValue kVal = inst.getCatVal(k);
//                const InstanceCount countOfxixjxk = xxxyDist_.getCount(noCatAtts_ - 1, tailVal, j, jVal, k, kVal);
//                const InstanceCount countOfxixj = xxxyDist_.xxyCounts.getCount(noCatAtts_ - 1, tailVal, j, jVal);
//                const InstanceCount countOfxixk = xxxyDist_.xxyCounts.getCount(noCatAtts_ - 1, tailVal, k, kVal);
//                const InstanceCount countOfxjxk = xxxyDist_.xxyCounts.getCount(j, jVal, k, kVal);
//                const InstanceCount countOfxj = xxxyDist_.xxyCounts.xyCounts.getCount(j, jVal);
//                const InstanceCount countOfxk = xxxyDist_.xxyCounts.xyCounts.getCount(k, kVal);
//                if (countOfxj == countOfxixj && countOfxj >= minCount) {
//                    //xj is a generalisation of xi
//                    generalizationSet[noCatAtts_] = true;
//                    //                                          fathercount[j]++;
//                    //                                                break;
//                } else if (countOfxk == countOfxixk && countOfxj != countOfxixj && countOfxk >= minCount) {
//                    //xj is a generalisation of xi
//                    generalizationSet[noCatAtts_] = true;
//                    //                                      fathercount[k]++;
//                } else if (countOfxjxk == countOfxixjxk && countOfxjxk >= minCount) {
//                    //xi is a generalisation or substitution of xj
//                    //once one xj has been found for xi, stop for rest j
//                    generalizationSet[noCatAtts_] = true;
//
//                }
//            }
//        }
    }


    for (CatValue y = 0; y < noClasses_; y++)
        classDist[y] = 0;

    // scale up by maximum possible factor to reduce risk of numeric underflow
   

    CatValue delta = 0;

    fdarray<double> spodeProbs(noCatAtts_, noClasses_);
    fdarray<InstanceCount> xyCount(noCatAtts_, noClasses_);
    std::vector<bool> active(noCatAtts_, false);

    //***********************************************************************************************       

    for (CatValue parent = 0; parent < noCatAtts_; parent++) {

        //discard the attribute that is not active or in generalization set
        if (!generalizationSet[parent]) {
            const CatValue parentVal = inst.getCatVal(parent);
            for (CatValue y = 0; y < noClasses_; y++) {
                spodeProbs[parent][y] = xxxyDist_.xxyCounts.xyCounts.jointP(parent, inst.getCatVal(parent), y);
            }
        }
    }

//    for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
//        for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
//            for (CatValue y = 0; y < noClasses_; y++) {
//                spodeProbs[x1][y] *= xxxyDist_.xxyCounts.jointP(x1, inst.getCatVal(x1), x2, inst.getCatVal(x2), y) / xxxyDist_.xxyCounts.xyCounts.jointP(x1, inst.getCatVal(x1), y);
//                spodeProbs[x2][y] *= xxxyDist_.xxyCounts.jointP(x1, inst.getCatVal(x1), x2, inst.getCatVal(x2), y) / xxxyDist_.xxyCounts.xyCounts.jointP(x2, inst.getCatVal(x2), y);
//            }
//        }
//    }

    for (CatValue y = 0; y < noClasses_; y++) {
        for (CategoricalAttribute parent = 0; parent < noCatAtts_; parent++) {
            for (CategoricalAttribute x1 = 0; x1 < noCatAtts_; x1++) {
                if ((x1 != parent)&&(effect_children[parent][x1]=true)) {                              
                   spodeProbs[parent][y] *= xxxyDist_.xxyCounts.jointP(x1, inst.getCatVal(x1), parent, inst.getCatVal(parent), y) / xxxyDist_.xxyCounts.xyCounts.jointP(parent, inst.getCatVal(parent), y);
                }
            }          
        }
    }

    for (CatValue parent = 0; parent < noCatAtts_; parent++) {       
            for (CatValue y = 0; y < noClasses_; y++) {
                classDist[y] += spodeProbs[parent][y];
            }       
    }
    //    float GenAttr = 0;
    //    for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
    //        if (!generalizationSet[i] == true && active_[i] == true)
    //            GenAttr++;
    //    }
    //    UsedAttrRatio += GenAttr / noCatAtts_;
    //    printf("UsedAttrRatio is %f,\n", UsedAttrRatio);
    //        float fathernor=0;
    //        for (CategoricalAttribute i = 0; i < noCatAtts_; i++) { 
    //            if(fathercount[i]==0){
    //                printf(" %d, ",i);
    //                fathernor++;
    //            }         
    // //         printf(" %d used %d times, ",i,fathercount[i]);
    //         }
    //        printf("%f never used,\n", static_cast<float>(fathernor)/noCatAtts_);

    normalise(classDist);
}

class valCmpClass {
public:

    valCmpClass(std::vector<float> *s) {
        val = s;
    }

    bool operator()(CategoricalAttribute a, CategoricalAttribute b) {
        return (*val)[a] > (*val)[b];
    }

private:
    std::vector<float> *val;
};

void aodeDouble::finalisePass() {
    assert(trainingIsFinished_ == false);

    crosstab<float> DoubleMI = crosstab<float>(noCatAtts_);
    crosstab<int> eff_paren = crosstab<int>(noCatAtts_);
    getDoubleMutualInf(xxxyDist_.xxyCounts, DoubleMI);


    printf("before sorting\n");
    for (int i = 0; i < noCatAtts_; i++) {
        for (int j = 0; j < noCatAtts_; j++) {

            printf(" %f", DoubleMI[i][j]);
        }
        printf("\n");
    }

    for (int i = 0; i < noCatAtts_; i++) {
        for (int j = 0; j < noCatAtts_; j++) {

            eff_paren[i][j] = j;
        }

    }
    //按照互信息排序，并选择排在前面的属性为父变量
    for (unsigned int i = 0; i < noCatAtts_; i++) {
        for (unsigned int j = 0; j < noCatAtts_; j++) {
            float bigger = DoubleMI[i][0];
            int flag = 0;
            for (unsigned int k = 0; k < noCatAtts_; k++) {
                if (bigger < DoubleMI[i][k]) {
                    bigger = DoubleMI[i][k];
                    flag = k;
                }
            }
            DoubleMI[i][flag] = -DoubleMI[i][flag];
            eff_paren[i][j] = flag;
        }
    }   
    
    //调整后的互信息次序  
    printf("after sorting\n");
    for (int i = 0; i < noCatAtts_; i++) {
        for (int j = 0; j < noCatAtts_; j++) {

            printf(" %f", DoubleMI[i][j]);
        }
        printf("\n");
    }
    //调整后的属性位置  
    printf("the new oder of attributes\n");
    for (int i = 0; i < noCatAtts_; i++) {
        for (int j = 0; j < noCatAtts_; j++) {

            printf(" %d", eff_paren[i][j]);
        }
        printf("\n");
    }
    //相对于父变量的条件互信息总和
    std::vector<float> sum_cmi;
    sum_cmi.assign(noCatAtts_, 0);
    for (int i = 0; i < noCatAtts_; i++) {
        for (int j = 0; j < noCatAtts_; j++) {
            if (j != i) {
                sum_cmi[i] += DoubleMI[i][j];
            }
        }
        float result = sum_cmi[i];
        printf("the sum of %d is %f,\n", i, result);
    }
    for (unsigned int i = 0; i < noCatAtts_; i++) {
        printf("the %d attrbute effect is ", i);
        float firstN_sum = 0;
        for (unsigned int j = 0; j < noCatAtts_; j++) {
            int k = eff_paren[i][j];
            if ((firstN_sum / sum_cmi[i]) < threshold) {
                firstN_sum += DoubleMI[i][k];
                effect_children[i][k] = true;
                printf("%d,", k);

            } else {
                break;
            }
        }
        printf("\n ");
    }
   

  
    //****************************************************************

    //    alpha = 0;
    //    printf("the father node is ");
    //    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
    //        if (active_[x1] == true) {
    //            printf(" %d ", x1);
    //            alpha++;
    //        }
    //    }
    //    printf(" \n ");
    //    printf("the whole number is %d, and the used number as parent is %d ", noCatAtts_, alpha);
    trainingIsFinished_ = true;
}

void aodeDouble::nbClassify(const instance &inst, std::vector<double> &classDist,
        xyDist &xyDist_) {

    for (CatValue y = 0; y < noClasses_; y++) {
        double p = xyDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);
        // scale up by maximum possible factor to reduce risk of numeric underflow

        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            p *= xyDist_.p(a, inst.getCatVal(a), y);
        }

        assert(p >= 0.0);
        classDist[y] = p;
    }
    normalise(classDist);
}

