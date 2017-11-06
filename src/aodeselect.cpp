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

aodeselect::aodeselect(char* const *& argv, char* const * end) {
    name_ = "AODE";
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

    
    //      else
    //        error("aode does not support argument %s", argv[0]);
    //    }

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

aodeselect::~aodeselect(void) {
}

void aodeselect::getCapabilities(capabilities &c) {
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void aodeselect::reset(InstanceStream &is) {
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

void aodeselect::initialisePass() {

}
class miCmpClass {
public:

    miCmpClass(std::vector<float> *m) {
        mi = m;
    }

    bool operator() (CategoricalAttribute a, CategoricalAttribute b) {
        return (*mi)[a] > (*mi)[b];
    }

private:
    std::vector<float> *mi;
};
void aodeselect::train(const instance &inst) {
    xxxyDist_.update(inst);
}

/// true iff no more passes are required. updated by finalisePass()

bool aodeselect::trainingIsFinished() {
    return trainingIsFinished_;
}

void aodeselect::classify(const instance &inst, std::vector<double> &classDist) {

    std::vector<bool> generalizationSet;
    const InstanceCount totalCount = xxxyDist_.xxyCounts.xyCounts.count;
    
   
    generalizationSet.assign(noCatAtts_, false);

    //compute the generalisation set and substitution set for
    //lazy subsumption resolution
    if (subsumptionResolution == true) {
        //********************************************************************************************            
        for (CategoricalAttribute i = 1; i < noCatAtts_ - 1; i++) {

            const CatValue iVal = inst.getCatVal(i);
            for (CategoricalAttribute j = 0; j < i; j++) {
                for (CategoricalAttribute k = i + 1; k < noCatAtts_; k++) {
                    //                               if (!generalizationSet[j] && !generalizationSet[k]) {
                    const CatValue jVal = inst.getCatVal(j);
                    const CatValue kVal = inst.getCatVal(k);
                    const InstanceCount countOfxixjxk = xxxyDist_.getCount(i, iVal, j, jVal, k, kVal);
                    const InstanceCount countOfxixj = xxxyDist_.xxyCounts.getCount(i, iVal, j, jVal);
                    const InstanceCount countOfxixk = xxxyDist_.xxyCounts.getCount(i, iVal, k, kVal);
                    const InstanceCount countOfxjxk = xxxyDist_.xxyCounts.getCount(j, jVal, k, kVal);
                    const InstanceCount countOfxj = xxxyDist_.xxyCounts.xyCounts.getCount(j, jVal);
                    const InstanceCount countOfxk = xxxyDist_.xxyCounts.xyCounts.getCount(k, kVal);
                    if (countOfxj == countOfxixj && countOfxj >= minCount) {
                        //xj is a generalisation of xi
                        //                     printf("%d to %d,", j, i);
                        generalizationSet[i] = true;

                    }
                    else if (countOfxk == countOfxixk && countOfxj != countOfxixj && countOfxk >= minCount) {
                        //xj is a generalisation of xi
                        //                        printf("%d to %d,", k, i);
                        generalizationSet[i] = true;

                    }
                    else if (countOfxjxk == countOfxixjxk && countOfxjxk >= minCount) {
                        //xi is a generalisation or substitution of xj
                        //once one xj has been found for xi, stop for rest j
                        //                       printf("%d and %d to %d,", j,k, i);
                        generalizationSet[i] = true;

                    }
                    //                                 }
                }
                //                               if (flagaode==true){
                //                                break;
                //                               }
            }

        }

        const CatValue headVal = inst.getCatVal(0);
        for (CategoricalAttribute j = 1; j < noCatAtts_ - 1; j++) {
            //	if (!generalizationSet[j]) {
            for (CategoricalAttribute k = j + 1; k < noCatAtts_; k++) {
                const CatValue jVal = inst.getCatVal(j);
                const CatValue kVal = inst.getCatVal(k);
                const InstanceCount countOfxixjxk = xxxyDist_.getCount(0, headVal, j, jVal, k, kVal);
                const InstanceCount countOfxixj = xxxyDist_.xxyCounts.getCount(0, headVal, j, jVal);
                const InstanceCount countOfxixk = xxxyDist_.xxyCounts.getCount(0, headVal, k, kVal);
                const InstanceCount countOfxjxk = xxxyDist_.xxyCounts.getCount(j, jVal, k, kVal);
                const InstanceCount countOfxj = xxxyDist_.xxyCounts.xyCounts.getCount(j, jVal);
                const InstanceCount countOfxk = xxxyDist_.xxyCounts.xyCounts.getCount(k, kVal);
                if (countOfxjxk == countOfxixjxk && countOfxjxk >= minCount) {
                    //xi is a generalisation or substitution of xj
                    //once one xj has been found for xi, stop for rest j
                    generalizationSet[0] = true;
                } else if (countOfxj == countOfxixj && countOfxj >= minCount) {
                    //	xj is a generalisation of xi
                    generalizationSet[0] = true;
                }
                else if (countOfxk == countOfxixk && countOfxk >= minCount) {
                    //xj is a generalisation of xi
                    generalizationSet[0] = true;
                }
            }
            //                                    break;
        }

        const CatValue tailVal = inst.getCatVal(noCatAtts_ - 1);
        for (CategoricalAttribute j = 0; j < noCatAtts_ - 2; j++) {
            //	if (!generalizationSet[j]) {
            for (CategoricalAttribute k = j + 1; k < noCatAtts_ - 1; k++) {
                const CatValue jVal = inst.getCatVal(j);
                const CatValue kVal = inst.getCatVal(k);
                const InstanceCount countOfxixjxk = xxxyDist_.getCount(noCatAtts_ - 1, tailVal, j, jVal, k, kVal);
                const InstanceCount countOfxixj = xxxyDist_.xxyCounts.getCount(noCatAtts_ - 1, tailVal, j, jVal);
                const InstanceCount countOfxixk = xxxyDist_.xxyCounts.getCount(noCatAtts_ - 1, tailVal, k, kVal);
                const InstanceCount countOfxjxk = xxxyDist_.xxyCounts.getCount(j, jVal, k, kVal);
                const InstanceCount countOfxj = xxxyDist_.xxyCounts.xyCounts.getCount(j, jVal);
                const InstanceCount countOfxk = xxxyDist_.xxyCounts.xyCounts.getCount(k, kVal);
                if (countOfxj == countOfxixj && countOfxj >= minCount) {
                    //xj is a generalisation of xi
                    generalizationSet[noCatAtts_] = true;
                    //                                          fathercount[j]++;
                    //                                                break;
                }
                else if (countOfxk == countOfxixk && countOfxj != countOfxixj && countOfxk >= minCount) {
                    //xj is a generalisation of xi
                    generalizationSet[noCatAtts_] = true;
                    //                                      fathercount[k]++;
                }
                else if (countOfxjxk == countOfxixjxk && countOfxjxk >= minCount) {
                    //xi is a generalisation or substitution of xj
                    //once one xj has been found for xi, stop for rest j
                    generalizationSet[noCatAtts_] = true;

                }

            }
            //                                    break;
        }


        //**************************************************************************************************************		

        if (verbosity >= 4) {
            for (CategoricalAttribute i = 0; i < noCatAtts_; i++)
                if (!generalizationSet[i])
                    printf("%d\t", i);
            printf("\n");
        }
    }

    if (verbosity >= 4) {
        for (CatValue i = 0; i < noCatAtts_; i++) {
            printf("%f\n", weight[i]);
        }
    }

    for (CatValue y = 0; y < noClasses_; y++)
        classDist[y] = 0;

    // scale up by maximum possible factor to reduce risk of numeric underflow
    double scaleFactor = std::numeric_limits<double>::max() / noCatAtts_;

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
                xyCount[parent][y] = xxxyDist_.xxyCounts.xyCounts.getCount(parent,
                        parentVal, y);
            }

            if (active_[parent]) { // to decide which attribute can be used as parent
                if (xxxyDist_.xxyCounts.xyCounts.getCount(parent, parentVal) > 0) { //calculate P(parent,y)
                    delta++; //decide if current attribute can be parent. if no attributes can be, then aode turned to be nb
                    active[parent] = true; // parentVal appears in training set, so it can be parent

                    if (empiricalMEst_) {
                        for (CatValue y = 0; y < noClasses_; y++) {
                            spodeProbs[parent][y] = weight[parent] //calculate P(parent,y)
                                    * empiricalMEstimate(xyCount[parent][y],
                                    totalCount,
                                    xxxyDist_.xxyCounts.xyCounts.p(y)
                                    * xxxyDist_.xxyCounts.xyCounts.p(
                                    parent, parentVal))
                                    * scaleFactor;
                        }
                    } else {
                        for (CatValue y = 0; y < noClasses_; y++) {
                            spodeProbs[parent][y] = weight[parent]
                                    * mEstimate(xyCount[parent][y], totalCount,
                                    noClasses_ * xxxyDist_.xxyCounts.getNoValues(parent))
                                    * scaleFactor;
                        }
                    }
                } else if (verbosity >= 5)
                    printf("%d\n", parent);
            }
        }
    }

    if (delta == 0) {
        nbClassify(inst, classDist, xxxyDist_.xxyCounts.xyCounts);
        return;
    }

    for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) { //calculate P(x1|x2,y)
        //
        //		std::vector<std::vector<std::vector<double> > > * parentsProbs =
        //				&xxyDist_.condiProbs[x1][x1Val];

        //discard the attribute that is in generalization set
        if (!generalizationSet[x1]) {
            const CatValue x1Val = inst.getCatVal(x1);
            const unsigned int noX1Vals = xxxyDist_.xxyCounts.getNoValues(x1);
            const bool x1Active = active[x1];
            constXYSubDist xySubDist(xxxyDist_.xxyCounts.getXYSubDist(x1, x1Val),
                    noClasses_);

            //calculate only for empricial2
            const InstanceCount x1Count = xxxyDist_.xxyCounts.xyCounts.getCount(x1,
                    x1Val);

            for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

                //	printf("c:%d\n", x2);
                if (!generalizationSet[x2] && effect_children[x1][x2]) {
                    const bool x2Active = active[x2];

                    if (x1Active || x2Active) {
                        CatValue x2Val = inst.getCatVal(x2);
                        const unsigned int noX2Vals = xxxyDist_.getNoValues(x2);

                        //calculate only for empricial2
                        InstanceCount x1x2Count = xySubDist.getCount(x2, x2Val, 0); //p(x1=x1val, x2=x2Val,y=0)
                        for (CatValue y = 1; y < noClasses_; y++) {
                            x1x2Count += xySubDist.getCount(x2, x2Val, y);
                        }//p(x1=x1val, x2=x2Val)
                        const InstanceCount x2Count =
                                xxxyDist_.xxyCounts.xyCounts.getCount(x2, x2Val);

                        const double pX2gX1 = empiricalMEstimate(x1x2Count, x1Count, xxxyDist_.xxyCounts.xyCounts.p(x2, x2Val)); //p(x2|x1)
                        const double pX1gX2 = empiricalMEstimate(x1x2Count, x2Count, xxxyDist_.xxyCounts.xyCounts.p(x1, x1Val)); //p(x1|x2)

                        for (CatValue y = 0; y < noClasses_; y++) {
                            const InstanceCount x1x2yCount = xySubDist.getCount(
                                    x2, x2Val, y);

                            if (x1Active) {
                                if (empiricalMEst_) {

                                    spodeProbs[x1][y] *= empiricalMEstimate(
                                            x1x2yCount, xyCount[x1][y],
                                            xxxyDist_.xxyCounts.xyCounts.p(x2, x2Val));
                                } else if (empiricalMEst2_) {
                                    //double probX2OnX1=mEstimate();
                                    spodeProbs[x1][y] *= empiricalMEstimate(
                                            x1x2yCount, xyCount[x1][y],
                                            pX2gX1);
                                } else {
                                    spodeProbs[x1][y] *= mEstimate(x1x2yCount,
                                            xyCount[x1][y], noX2Vals);
                                }
                            }
                            if (x2Active) {
                                if (empiricalMEst_) {
                                    spodeProbs[x2][y] *= empiricalMEstimate(
                                            x1x2yCount, xyCount[x2][y],
                                            xxxyDist_.xxyCounts.xyCounts.p(x1, x1Val));
                                } else if (empiricalMEst2_) {
                                    //double probX2OnX1=mEstimate();
                                    spodeProbs[x1][y] *= empiricalMEstimate(
                                            x1x2yCount, xyCount[x1][y],
                                            pX1gX2);
                                } else {
                                    spodeProbs[x2][y] *= mEstimate(x1x2yCount,
                                            xyCount[x2][y], noX1Vals);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (CatValue parent = 0; parent < noCatAtts_; parent++) {
        if (active_[parent]) {
            for (CatValue y = 0; y < noClasses_; y++) {
                classDist[y] += spodeProbs[parent][y];
            }
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

// creates a comparator for two attributes based on their
//relative value with the class,such as mutual information, symmetrical uncertainty

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

void aodeselect::finalisePass() {
    assert(trainingIsFinished_ == false);

    crosstab<float> cmi = crosstab<float>(noCatAtts_);
    crosstab<int> eff_paren = crosstab<int>(noCatAtts_);
    getCondMutualInf(xxxyDist_.xxyCounts, cmi);
       printf("before sorting\n");
    for (int i = 0; i < noCatAtts_; i++) {
        for (int j = 0; j < noCatAtts_; j++) {

            printf(" %f", cmi[i][j]);
        }
        printf("\n");
    }

    for (int i = 0; i < noCatAtts_; i++) {
        for (int j = 0; j < noCatAtts_; j++) {

            eff_paren[i][j] = j;
        }

    }
   //按照条件互信息排序，并选择排在前面的属性为父变量
    for (unsigned int i = 0; i < noCatAtts_; i++) {
               for (unsigned int j = 0; j < noCatAtts_; j++) {
                     float bigger=cmi[i][0];
                     int flag=0;
                     for (unsigned int k = 0; k < noCatAtts_; k++) {
                        if (bigger<cmi[i][k]){
                             bigger=cmi[i][k];
                            flag=k;
                         }                   
                      } 
                     cmi[i][flag]=-cmi[i][flag];	               
                     eff_paren[i][j]=flag;
           }
      }

    //调整后的互信息次序  
    printf("after sorting\n");
    for (int i = 0; i < noCatAtts_; i++) {
        for (int j = 0; j < noCatAtts_; j++) {

            printf(" %f", cmi[i][j]);
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
    sum_cmi.assign(noCatAtts_,0);
      for (int i = 0; i < noCatAtts_; i++) {
          for (int j = 0; j < noCatAtts_ ; j++) {
             if (j!=i){
                 sum_cmi[i] += cmi[i][j];
             }            
      }
          sum_cmi[i]=-sum_cmi[i];
       float result=sum_cmi[i];
       printf("the sum of %d is %f,\n", i, result);
       
    }
    float sum_all=0;
    for (unsigned int i = 0; i < noCatAtts_; i++) {
        sum_all+=sum_cmi[i];
    } 
    miCmpClass cmp(&sum_cmi);
    std::vector<CategoricalAttribute> order;
    order.clear();
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        order.push_back(a);
    }
    std::sort(order.begin(), order.end(), cmp);
    printf("order of sum_cmi\n");
    for (std::vector<CategoricalAttribute>::const_iterator i = order.begin(); i != order.end(); i++) {
        unsigned int u = *i;
        printf("sum_cmi %u= %f\n", u,sum_cmi[u]);
    }
    printf("the ratio of parent  is "); 
    float firstN_sum = 0; 
    float a=0;
    for (std::vector<CategoricalAttribute>::const_iterator i = order.begin(); i != order.end(); i++) {
        unsigned int u = *i;       
        if (a / sum_all < 0.9) {
            a+=sum_cmi[u];
            firstN_sum++;  
            active_[u] = true;
        }  else{
            break;
        }     
    }
    a=1-firstN_sum/noCatAtts_;
     printf("%f\n", a); 
        
    
    //**************************************************************************************************************************
    // find the maximum spanning tree

    CategoricalAttribute firstAtt = 0;

    parents_[firstAtt] = NOPARENT;

    float *maxWeight;
    CategoricalAttribute *bestSoFar;
    CategoricalAttribute topCandidate = firstAtt;
    std::set<CategoricalAttribute> available;

    safeAlloc(maxWeight, noCatAtts_);
    safeAlloc(bestSoFar, noCatAtts_);

    maxWeight[firstAtt] = -std::numeric_limits<float>::max();

    for (CategoricalAttribute a = firstAtt + 1; a < noCatAtts_; a++) {
        maxWeight[a] = cmi[firstAtt][a];
        if (cmi[firstAtt][a] > maxWeight[topCandidate])
            topCandidate = a;
        bestSoFar[a] = firstAtt;
        available.insert(a);
    }

    while (!available.empty()) {
        const CategoricalAttribute current = topCandidate;
        parents_[current] = bestSoFar[current];
        available.erase(current);

        if (!available.empty()) {
            topCandidate = *available.begin();
            for (std::set<CategoricalAttribute>::const_iterator it =
                    available.begin(); it != available.end(); it++) {
                if (maxWeight[*it] < cmi[current][*it]) {
                    maxWeight[*it] = cmi[current][*it];
                    bestSoFar[*it] = current;
                }

                if (maxWeight[*it] > maxWeight[topCandidate])
                    topCandidate = *it;
            }
        }
    }


    //delete []mi;
    delete[] bestSoFar;
    delete[] maxWeight;



    unsigned int alpha = 0;
//    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
//
//        const CategoricalAttribute parent = parents_[x1];
//        if (parent != NOPARENT) {
//            active_[parent] = true;
//        }
//    }
    //*************************************************************************
    //         for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
    //             alpha=0;
    //             if (active_[x1] == true){
    //                 for (unsigned int x2 = 0; x2 < noCatAtts_; x2++) {
    //                   const CategoricalAttribute parent = parents_[x1];
    //                   if (parent == x2) {
    //		       alpha++;      
    //                   }     
    //                 }
    //                 if(alpha>0){
    //                     active_[x1] = true;   
    //                  }
    //                 else{
    //                   active_[x1] = false;    
    //                  }
    //               }              
    //            }
    //****************************************************************

    alpha = 0;
 //   printf("the father node is ");
    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        if (active_[x1] == true) {
    //        printf(" %d ", x1);
            alpha++;
        }
    }
 //   printf(" \n ");
 //   printf("the whole number is %d, and the used number as parent is %d ", noCatAtts_, alpha);
    trainingIsFinished_ = true;
}

void aodeselect::nbClassify(const instance &inst, std::vector<double> &classDist,
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

