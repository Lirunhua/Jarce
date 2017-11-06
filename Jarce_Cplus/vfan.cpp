/* Open source system for classification learning from very large data
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
 ** Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
 */
#include "vfan.h"
#include "utils.h"
#include "correlationMeasures.h"
#include <assert.h>
#include <math.h>
#include <set>
#include <stdlib.h>

vfan::vfan() :
		trainingIsFinished_(false) {
}

vfan::vfan(char* const *&, char* const *) :
		xxxyDist_(), trainingIsFinished_(false) {
	name_ = "TAN";
}

vfan::~vfan(void) {}

void vfan::reset(InstanceStream &is) {
	instanceStream_ = &is;
	const unsigned int noCatAtts = is.getNoCatAtts();
	noCatAtts_ = noCatAtts;
	noClasses_ = is.getNoClasses();

	trainingIsFinished_ = false;

	//safeAlloc(parents, noCatAtts_);
        parents_.resize(noCatAtts);
	for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
		parents_[a] = NOPARENT;
	}
        active_.assign(noCatAtts_, false);
	xxxyDist_.reset(is);
}

void vfan::getCapabilities(capabilities &c) {
	c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void vfan::initialisePass() {
	assert(trainingIsFinished_ == false);
	//learner::initialisePass (pass_);
//	dist->clear();
//	for (CategoricalAttribute a = 0; a < meta->noCatAtts; a++) {
//		parents_[a] = NOPARENT;
//	}
}

void vfan::train(const instance &inst) {
	xxxyDist_.update(inst);
}

void vfan::classify(const instance &inst, std::vector<double> &classDist) {

	for (CatValue y = 0; y < noClasses_; y++) {
		classDist[y] = xxxyDist_.xxyCounts.xyCounts.p(y);
       
	}
       
     float Pxy;        
//****************************************************************************important for vfan
     for (CatValue y = 0; y < noClasses_; y++) {
        float SUM=1;
	for (unsigned int i = 1; i < noCatAtts_; i++) {
	   int headattr=order[i];
           Pxy=xxxyDist_.xxyCounts.xyCounts.p(headattr,inst.getCatVal(headattr),y);  //P(Xi|Y))
           int otherattr;
           for (unsigned int j = 0; j < i; j++) {
             otherattr=order[j];
             if (active_[otherattr]==true){
             Pxy=Pxy+xxxyDist_.xxyCounts.jointP(headattr,inst.getCatVal(headattr),otherattr,inst.getCatVal(otherattr),y)/xxxyDist_.xxyCounts.xyCounts.jointP(otherattr,inst.getCatVal(otherattr),y);
             }
           }// SUM OF P(Xi|C,Xj)
           int otherj, otherk;
           for (unsigned int j = 1; j < i; j++) {
               for (unsigned int k = 0; k < j; k++) {
                    otherj=order[j];
                    otherk=order[k];
                    Pxy=Pxy+xxxyDist_.jointP(headattr,inst.getCatVal(headattr),otherj,inst.getCatVal(otherj),otherk,inst.getCatVal(otherk),y)/xxxyDist_.xxyCounts.jointP(otherj,inst.getCatVal(otherj),otherk,inst.getCatVal(otherk),y);
               }
            
           }// SUM OF P(Xi|C,Xj,Xk)
             SUM*=Pxy;
             SUM=SUM/(i*i+i);
    	}
        SUM*=xxxyDist_.xxyCounts.xyCounts.p(order[0],inst.getCatVal(order[0]),y); //SUM*P(X0|Y))
        classDist[y]=SUM*classDist[y];
     }
//**************************************************************************
//        for (CatValue y = 0; y < noClasses_; y++) {
//         float SUM=0;
//         for (unsigned int i = 1; i < noCatAtts_-1; i++) {
//	     SUM=xxxyDist_.xxyCounts.xyCounts.jointP(i,inst.getCatVal(i),y)/xxxyDist_.xxyCounts.xyCounts.p(i,inst.getCatVal(i));  //P(Y|Xi)
//             for (unsigned int j = 0; j < i; j++) {
//               SUM+=xxxyDist_.xxyCounts.jointP(i,inst.getCatVal(i),j,inst.getCatVal(j),y)/xxxyDist_.xxyCounts.jointP(i,inst.getCatVal(i),j,inst.getCatVal(j)); 
//              }// SUM OF P(C|Xi,Xj)
//             for (unsigned int j = i+1; j < noCatAtts_; j++) {
//             SUM+=xxxyDist_.xxyCounts.jointP(i,inst.getCatVal(i),j,inst.getCatVal(j),y)/xxxyDist_.xxyCounts.jointP(i,inst.getCatVal(i),j,inst.getCatVal(j)); 
//              }// SUM OF P(C|Xi,Xj)
//          }
//          for (unsigned int j = 1; j < noCatAtts_; j++) {
//               SUM=xxxyDist_.xxyCounts.xyCounts.jointP(0,inst.getCatVal(0),y)/xxxyDist_.xxyCounts.xyCounts.p(0,inst.getCatVal(0));  //P(Y|X0)
//               SUM+=xxxyDist_.xxyCounts.jointP(0,inst.getCatVal(0),j,inst.getCatVal(j),y)/xxxyDist_.xxyCounts.jointP(0,inst.getCatVal(0),j,inst.getCatVal(j)); 
//              }// SUM OF P(C|Xi,Xj)
//          for (unsigned int j = 0; j < noCatAtts_-1; j++) {
//               SUM=xxxyDist_.xxyCounts.xyCounts.jointP(noCatAtts_-1,inst.getCatVal(noCatAtts_-1),y)/xxxyDist_.xxyCounts.xyCounts.p(noCatAtts_-1,inst.getCatVal(noCatAtts_-1));  //P(Y|Xi)
//               SUM+=xxxyDist_.xxyCounts.jointP(noCatAtts_-1,inst.getCatVal(noCatAtts_-1),j,inst.getCatVal(j),y)/xxxyDist_.xxyCounts.jointP(noCatAtts_-1,inst.getCatVal(noCatAtts_-1),j,inst.getCatVal(j)); 
//              }// SUM OF P(C|Xi,Xj)
//          classDist[y]=SUM;
     //     printf("%f, ",SUM);
//        }  
//    	
        normalise(classDist);
}

void vfan::finalisePass() {
	assert(trainingIsFinished_ == false);

	//// calculate conditional mutual information
	//float **mi = new float *[meta->noAttributes];

	//for (attribute a = 0; a < meta->noAttributes; a++) {
	//  mi[a] = new float[meta->noAttributes];
	//}

	//const double totalCount = dist->xyCounts.count;

	//for (attribute x1 = 1; x1 < meta->noAttributes; x1++) {
	//  if (meta->attTypes[x1] == categorical) {
	//    for (attribute x2 = 0; x2 < x1; x2++) {
	//      if (meta->attTypes[x2] == categorical) {
	//        float m = 0.0;

	//        for (cat_value v1 = 0; v1 < meta->noValues[x1]; v1++) {
	//          for (cat_value v2 = 0; v2 < meta->noValues[x2]; v2++) {
	//            for (unsigned int y = 0; y < meta->noClasses(); y++) {
	//              const double x1x2y = dist->getCount(x1, v1, x2, v2, y);
	//              if (x1x2y) {
	//                //const unsigned int yCount = dist->xyCounts.getClassCount(y);
	//                //const unsigned int  x1y = dist->xyCounts.getCount(x1, v1, y);
	//                //const unsigned int  x2y = dist->xyCounts.getCount(x2, v2, y);
	//                m += (x1x2y/totalCount) * log(dist->xyCounts.getClassCount(y) * x1x2y / (static_cast<double>(dist->xyCounts.getCount(x1, v1, y))*dist->xyCounts.getCount(x2, v2, y)));
	//                //assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision
	//              }
	//            }
	//          }
	//        }

	//        assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision
	//        mi[x1][x2] = m;
	//        mi[x2][x1] = m;
	//      }
	//    }
	//  }
	//}

	std::vector<float> measure;
	getMutualInformation(xxxyDist_.xxyCounts.xyCounts, measure);
//        for (CatValue y = 0; y < noClasses_; y++) {
//				classDist[y] *= xxyDist_.xyCounts.p(x1, inst.getCatVal(x1), y);
//			}
//        for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
//		printf("%f,",measure[x1]);		
//	}
//       printf("\n");
      
        
        
//         for (unsigned int j = 0; j < noCatAtts_; j++) {
//		printf("%f,",measure[j]);		
//	}
//        printf("\n");
         for (unsigned int j = 0; j < noCatAtts_; j++) {
                float bigger=measure[0];
                int flag=0;
                for (unsigned int i = 0; i < noCatAtts_; i++) {
                        if (bigger<measure[i]){
                             bigger=measure[i];
                            flag=i;
                             }                     
                 } 
//               if (bigger==0){
//                   order[j]=j;   
//                   measure[0]=-1;
//               }
               measure[flag]=-1;	               
               order[j]=flag;
           }
           
        crosstab<float> cmi = crosstab<float>(noCatAtts_);
 	getCondMutualInf(xxxyDist_.xxyCounts, cmi);
  //    getproCondMutualInf(xxyDist_, cmi);
        
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
        
         unsigned int alpha=0;
         for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
              
              const CategoricalAttribute parent = parents_[x1];
               if (parent != NOPARENT) {
		    active_[parent] = true;	
             }             
   	}
	trainingIsFinished_ = true;
}

/// true iff no more passes are required. updated by finalisePass()
bool vfan::trainingIsFinished() {
	return trainingIsFinished_;
}
