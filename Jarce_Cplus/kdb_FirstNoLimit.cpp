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

#include "kdb_FirstNoLimit.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

kdb_FirstNoLimit::kdb_FirstNoLimit() : pass_(1) {
}

kdb_FirstNoLimit::kdb_FirstNoLimit(char*const*& argv, char*const* end) : pass_(1) {
    name_ = "kdb_FirstNoLimit";        
    k_ = 1;
    threshold = 0.9;
   
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

kdb_FirstNoLimit::~kdb_FirstNoLimit(void) {
}

void kdb_FirstNoLimit::getCapabilities(capabilities &c) {
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

void kdb_FirstNoLimit::reset(InstanceStream &is) {
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();
    sum_cmi.assign(noCatAtts_,0);
    k_ = min(k_, noCatAtts_ - 1); // k cannot exceed the real number of categorical attributes - 1

    // initialise distributions
    dTree_.resize(noCatAtts);
    parents_.resize(noCatAtts);

    for (CategoricalAttribute a = 0; a < noCatAtts; a++) {
        parents_[a].clear();
        dTree_[a].init(is, a);
    }

    dist_.reset(is);
    
    classDist_.reset(is);

    pass_ = 1;
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass

void kdb_FirstNoLimit::train(const instance &inst) {
    if (pass_ == 1) {
        dist_.update(inst);
    } else {
        assert(pass_ == 2);

        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            dTree_[a].update(inst, a, parents_[a]);
        }
        classDist_.update(inst);
    }
}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void kdb_FirstNoLimit::initialisePass() {
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void kdb_FirstNoLimit::finalisePass() {


    if (pass_ == 1) {
        // calculate the mutual information from the xy distribution
        std::vector<float> mi;
        getMutualInformation(dist_.xyCounts, mi);     
        
        crosstab<float> cmi = crosstab<float>(noCatAtts_);
        getCondMutualInf(dist_, cmi);

        // calculate the conditional mutual information from the xxy distribution

        dist_.clear();       

        // sort the attributes on MI with the class
        std::vector<CategoricalAttribute> order;
        
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            order.push_back(a);
        }
        // assign the parents
        if (!order.empty()) {
    
           for (std::vector<CategoricalAttribute>::const_iterator i = order.begin(); i != order.end(); i++) {                
                for (std::vector<CategoricalAttribute>::const_iterator j = order.begin(); j != order.end(); j++) {
                    float k=cmi[*i][*j];
                    printf("%u,%u is %f;", *i,*j, k);
                }
                 printf("\n");
            }
             //打印互信息次序
            for (std::vector<CategoricalAttribute>::const_iterator it = order.begin() ; it != order.end(); it++) {
                float k = mi[*it];
                printf("the MI of %u is %f\n", *it, k);

            }           
            for (std::vector<CategoricalAttribute>::const_iterator it = order.begin(); it != order.end(); it++) {                
                for (std::vector<CategoricalAttribute>::const_iterator i = order.begin(); i != order.end(); i++) {   
                    cmi[*it][*it] += cmi[*it][*i];
                }
                //把互信息的和放在下标为0的矩阵内
                float k=cmi[*it][*it];  
                sum_cmi[*it]=k;  
                printf("the sum of CMI of %u is %f\n", *it, k);

            }
// //*******************************************************************************            
            miCmpClass cmp(&mi);
            //根据互信息大小将属性排序并存储到order中   
            std::sort(order.begin(), order.end(), cmp);
//*******************************************************************************        
//            miCmpClass cmp(&sum_cmi);
//            //根据互信息总合大小将属性排序并存储到order中   
//            std::sort(order.begin(), order.end(), cmp);
 /*************************************************************************************************************/         
  //           proper KDB assignment of parents
            for (std::vector<CategoricalAttribute>::const_iterator it = order.begin() + 1; it != order.end(); it++) {
                parents_[*it].push_back(order[0]); //把前k个变量预先作为父变量，0号变量必在其中
                for (std::vector<CategoricalAttribute>::const_iterator it2 = order.begin() + 1; it2 != it; it2++) {
                    // make parents into the top k attributes on mi that precede *it in order
                    if (parents_[*it].size() < k_) {//如果父变量数目<k,直接以前几个变量为父变量；通过下面循环，使得父变量集合初始化为前几个变量
                        // create space for another parent
                        // set it initially to the new parent.
                        // if there is a lower value parent, the new parent will be inserted earlier and this value will get overwritten
                        parents_[*it].push_back(*it2);
                    }
                    //当新的变量it2加入到候选父变量集合内，计算它与it的条件互信息，并与现有的条件互信息比较，如果较大，则替代该父变量
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

           


//
            printf("the first one is %u\n", *order.begin());              

// //         找到信息量占有率低于给定阈值的属性
//          for (std::vector<CategoricalAttribute>::const_iterator it = CMI_order.begin(); it != CMI_order.end(); it++) {
//                if ((firstN_sum / sum_cmi) < threshold) {
//                    if (*it != *order.begin()) {
//                        firstN_sum += DoubleMI[0][*it]; 
//                        printf("%u 's sub_CMI is %f, ", *it,firstN_sum);
//                        eff_children[*it]=true;                    
//                    }
//                }
//            }   
//             for (std::vector<CategoricalAttribute>::const_iterator it = CMI_order.begin(); it != CMI_order.end(); it++) {               
//                    if (eff_children[*it]!=true) {
//                        parents_[*it].clear();;                       
//                    }
//                }
//                        
//              
            printf("\n ");
            for (std::vector<CategoricalAttribute>::const_iterator it = order.begin(); it != order.end(); it++) {  
                printf("%u 's parent is  ", *it);
                for (unsigned int i = 0; i < parents_[*it].size(); i++) {
                    unsigned int j= parents_[*it][i];
                     printf(" %u, ",  j);
                }
                printf("\n");
            }            
        }
        //************************************************************
    }

    ++pass_;

}
/// true iff no more passes are required. updated by finalisePass()

bool kdb_FirstNoLimit::trainingIsFinished() {
    return pass_ > 2;
}

void kdb_FirstNoLimit::classify(const instance& inst, std::vector<double> &posteriorDist) {
    // calculate the class probabilities in parallel
    // P(y)
    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0); // scale up by maximum possible factor to reduce risk of numeric underflow
    }

    // P(x_i | x_p1, .. x_pk, y)
    for (CategoricalAttribute x = 0; x < noCatAtts_; x++) {
 //      if (eff_children[x] == true) {
            dTree_[x].updateClassDistribution(posteriorDist, x, inst);
//        }       
    }

    // normalise the results
    normalise(posteriorDist);
}



