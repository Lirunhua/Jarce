/* Open source system for classification learning from very large data
 ** Copyright (C) 2012 Geoffrey I Webb
 ** Implements Sahami's k-dependence Bayesian classifier
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
 ** Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
 */
#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>

#include "kdbOri.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

kdbOri::kdbOri() : pass_(1) {
}

kdbOri::kdbOri(char*const*& argv, char*const* end) : pass_(1) {
    name_ = "KDBOri";
    k_ =1;
    threshold = 0.95;

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

kdbOri::~kdbOri(void) {
}

void kdbOri::getCapabilities(capabilities &c) {
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

// creates a comparator for two attributes based on their relative mutual information with the class

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

void kdbOri::reset(InstanceStream &is) {
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();
    sum_cmi.assign(noCatAtts_, 0);
    k_ = min(k_, noCatAtts_ - 1); // k cannot exceed the real number of categorical attributes - 1

    // initialise distributions
    dTree_.resize(noCatAtts);
    parents_.resize(noCatAtts);

    SUM_Wij.resize(noCatAtts);
    Wij.resize(noCatAtts);
    seq.resize(noCatAtts);
    for (CategoricalAttribute a = 0; a < noCatAtts; a++) {
        Wij[a].resize(noCatAtts);
    }

    for (CategoricalAttribute a = 0; a < noCatAtts; a++) {
        parents_[a].clear();
        dTree_[a].init(is, a);
    }

    dist_.reset(is);

    classDist_.reset(is);

    pass_ = 1;
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass

void kdbOri::train(const instance &inst) {
    if (pass_ == 1) {
        dist_.update(inst);
    } else {
        assert(pass_ == 2);


        //        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        //            dTree_[a].update(inst, a, parents_[a]);
        //        }
        classDist_.update(inst);
    }
}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void kdbOri::initialisePass() {
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void kdbOri::finalisePass() {


    if (pass_ == 1) {
        // calculate the mutual information from the xy distribution
    
        
        
        std::vector<float> mi;
        getMutualInformation(dist_.xyCounts, mi);

        crosstab<float> cmi = crosstab<float>(noCatAtts_);
        getCondMutualInf(dist_, cmi);

        crosstab<float> DoubleMI = crosstab<float>(noCatAtts_);
        getXCondMutualInf(dist_, DoubleMI);

        // calculate the conditional mutual information from the xxy distribution

        //      dist_.clear();

        // sort the attributes on MI with the class
        std::vector<CategoricalAttribute> order;
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            order.push_back(a);
        }

        // assign the parents
        if (!order.empty()) {
            printf("the value of conditional mutual information is as follows \n");
            for (std::vector<CategoricalAttribute>::const_iterator i = order.begin(); i != order.end(); i++) {
                for (std::vector<CategoricalAttribute>::const_iterator j = order.begin(); j != order.end(); j++) {
                    float k = cmi[*i][*j];
                    printf("%u,%u is %f;", *i, *j, k);
                }
                printf("\n");
            }


            //打印互信息次序
            miCmpClass cmp(&mi);
            std::sort(order.begin(), order.end(), cmp);
            printf("order of MI\n");


//sequence是根据条件互信息大小的排序结果
            int sequence = 0;
            for (std::vector<CategoricalAttribute>::const_iterator it = order.begin(); it != order.end(); it++) {
                seq[sequence] = *it;
                sequence++;
            }
            for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
                unsigned v = seq[a];
                float mi_val=mi[v];
                printf("the %d is %u, value is %f;\n ", a, v, mi_val);
            }



            //计算每个属性的条件互信息综合
            for (unsigned int i = 1; i < noCatAtts_; i++) {
                unsigned int k = seq[i];
                sum_cmi[k] = 0;
                for (unsigned int j = 0; j < i; j++) {
                    unsigned int l = seq[j];
                    sum_cmi[k] += cmi[k][l];
//                   float w= cmi[k][l];
//                   printf("%u,%u, %f;;;",k,l,w);
                }
            }
            //计算每个属性的权重
     //       printf("The weight of different attributes are:\n");
            for (unsigned int i = 1; i < noCatAtts_; i++) {
                unsigned int k = seq[i];
       //         printf("parent is: %u,  ", k);
                for (unsigned int j = 0; j < i; j++) {
                    unsigned int l = seq[j];
       //             printf("child is %u; ", l);
                    if (sum_cmi[k] > 0) {
                        Wij[k][l] = cmi[k][l] / sum_cmi[k];
                    }
                    else{
                        Wij[k][l] =0;
                    }
                    float uv= Wij[k][l];
        //            printf("weight is: %f; ", uv);
                }
                printf("\n ");
            }
            for (CatValue y = 0; y < noClasses_; y++) {
                double PX[noCatAtts_];
                for (unsigned int i = 1; i < noCatAtts_; i++) {                    
                    unsigned int k = seq[i]; //第i个位置的属性，由于从第1个属性开始有父节点，所以i从1开始
 //                   printf("when y is %d, the parent of %u is ", y, k);
                    PX[k] = 0;
                    if (sum_cmi[k] > 0) {
                        for (unsigned int j = 0; j <i; j++) {//j是i前面的属性，即i的父变量
                            unsigned int l = seq[j];
 //                           printf("%u,", l);
                        }
                    }
 //                   printf("\n************\n ");
                }
            }

        }
    }
    ++pass_;

}
/// true iff no more passes are required. updated by finalisePass()

bool kdbOri::trainingIsFinished() {
    return pass_ > 2;
}

void kdbOri::classify(const instance& inst, std::vector<double> &posteriorDist) {
    classDist_.update(inst);
//重新确定属性次序************************************************************
//        std::vector<CategoricalAttribute> order;
//    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
//        order.push_back(a);
//    }
//    std::vector<float> mi;
//    getMutualInformation(dist_.xyCounts, mi);
//    miCmpClass cmp(&mi);
//    std::sort(order.begin(), order.end(), cmp);
//    
//
//    int sequence = 0;
//    for (std::vector<CategoricalAttribute>::const_iterator it = order.begin(); it != order.end(); it++) {
//        seq[sequence] = *it;
//        sequence++;
//    }
//*************************************************************************    
    // calculate the class probabilities in parallel
    // P(y)
    for (CatValue y = 0; y < noClasses_; y++) {
        int i = seq[0];
        posteriorDist[y] = dist_.xyCounts.jointP(i, inst.getCatVal(i), y);           
       
    }   
    //***********************************************
    for (CatValue y = 0; y < noClasses_; y++) {
        double PX[noCatAtts_];
        for (unsigned int i = 1; i < noCatAtts_; i++) {
            unsigned int k = seq[i]; //第i个位置的属性，由于从第1个属性开始有父节点，所以i从1开始            
            PX[k] = 0;
            if (sum_cmi[k] > 0) {                
                for (unsigned int j = 0; j < i; j++) {//j是i前面的属性，即i的父变量
                    unsigned int l = seq[j];                   
                    float jointPX1X2Y = dist_.jointP(k, inst.getCatVal(k), l, inst.getCatVal(l), y);
                    float jointPX2Y = dist_.xyCounts.jointP(l, inst.getCatVal(l), y);
                    PX[k] += Wij[k][l] * jointPX1X2Y / jointPX2Y;                   
                }
            } else {
                PX[k] = dist_.xyCounts.p(k, inst.getCatVal(k), y);
            }            
            posteriorDist[y] *= PX[k];
        }       
    }


    // normalise the results
    normalise(posteriorDist);


}



