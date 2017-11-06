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
 ** Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
 */

#include "crosstab.h"
#include "crosstab3D.h"
#include "xyDist.h"
#include "xxyDist.h"
#include "xxxyDist.h"
#include "xxxxyDist.h"
#include "yDist.h"
#include "math.h"
/**
<!-- globalinfo-start -->
 * File that includes different correlation measures among variables.<br/>
 <!-- globalinfo-end -->
 *
 * @author Geoff Webb (geoff.webb@monash.edu)
 * @author Ana M. Martinez (anam.martinez@monash.edu)
 */

double getEntropy(std::vector<InstanceCount>& classDist);

/**
 * Calculates the information gain between an attribute in dist and the class
 * 
 * @param dist  counts for the xy distributions.
 * @param a the attribute.
 */
double getInfoGain(xyDist& dist, CategoricalAttribute a);

/**
 * Calculates the information of an attribute
 * 
 * @param dist  counts for the xy distributions.
 * @param a the attribute.
 */
double getInformation(xyDist& dist, CategoricalAttribute a);

/**
 * Calculates the gain ratio between the attribute and the
 * class
 * 
 * @param dist  counts for the xy distributions.
 * @param a the attribute.
 */
double getGainRatio(xyDist& dist, CategoricalAttribute a);

/**
 * Calculates the gain ratio for a potential split class distribution
 * 
 * @param left the distribution for the first branch
 * @param right the distribution for the second branch
 */
double getGainRatio(std::vector<InstanceCount> &left, std::vector<InstanceCount> &right);

/**
 * Calculates the gain ratio for a potential split class distribution
 * 
 * @param left the distribution for the first branch
 * @param right the distribution for the second branch
 * @param unknown the distribution for instances with missing values
 */
double getGainRatio(std::vector<InstanceCount> &left, std::vector<InstanceCount> &right, std::vector<InstanceCount> &unknown);

/**
 * Calculates the mutual information between the attributes in dist and the
 * class
 * 
 * MI(X;C) = H(X) - H(X|C)
 * 
 * @param dist  counts for the xy distributions.
 * @param[out] mi mutual information between the attributes and the class.
 */

double H_General(xxyDist &dist, CategoricalAttribute xi, CategoricalAttribute xj, int q);
double H_loc(xxyDist &dist, CategoricalAttribute xi, CategoricalAttribute xj, int q, const instance & inst);
void getH_xy(xyDist &dist, std::vector<float> &mi);
void getlocH_xy(xyDist &dist, std::vector<float> &mi, const instance & inst);
void displayClassify(yDist &classDist_, xxxyDist & dist_1, std::vector<std::vector<CategoricalAttribute> > &parents_1, const instance & inst);

//基于nb,k=2,测度为p*log(p/m)
double H_standard_rewrite(yDist &classDist, xxxyDist &dist, std::vector<std::vector<CategoricalAttribute> > &parents, const instance & inst);
//非nb,k=2,测度为p*log(p/m)
double H_standard_rewrite_only_have_parents(yDist &classDist, xxxyDist &dist, CategoricalAttribute x0, std::vector<std::vector<CategoricalAttribute> > &parents, const instance & inst);


//基于nb,k=2,测度为(p/m)*log(p/m)
double H_standard_loc_k2(yDist &classDist, xxxyDist &dist, std::vector<std::vector<CategoricalAttribute> > &parents, const instance & inst);
void show_p(yDist &classDist, xxxyDist &dist, std::vector<std::vector<CategoricalAttribute> > &parents, CategoricalAttribute xk, const instance & inst);
double re_show_p(yDist &classDist, xxxyDist &dist, std::vector<std::vector<CategoricalAttribute> > &parents, const instance & inst);
//非nb,k=2,测度为(p/m)*log(p/m)
double H_standard_loc_k2_only_have_parents(yDist &classDist, xxxyDist &dist, std::vector<CategoricalAttribute> &parentIsC_order, std::vector<std::vector<CategoricalAttribute> > &parents, const instance & inst);

//非nb(成长型)，为属性找父节点过程，max{P(x1,x2,...,xn)} * max{H(Y) - H(Y|x1,x2,...,xn)}, H测度为(p/m)*log(p/m)
//x0为根，temp为候选父结点,parents为已生成的结构
void find_parents_PH_noder(yDist &classDist, xxxyDist & dist, CategoricalAttribute x0, CategoricalAttribute x, std::vector<CategoricalAttribute> &temp, std::vector<std::vector<CategoricalAttribute> > &parents,
        std::vector<CategoricalAttribute> &order, const instance & inst);
//P(x1,x2,...,xn)*{H(Y) - H(Y|x1,x2,...,xn)}
double H_PH_only_have_parents(yDist &classDist, xxxyDist &dist, CategoricalAttribute x0, std::vector<std::vector<CategoricalAttribute> > &parents, const instance & inst);
double H_PH(yDist &classDist, xxxyDist &dist, std::vector<std::vector<CategoricalAttribute> > &parents, const instance & inst);

//非nb,k=2,找父节点过程
void findmink_k2_only_have_parents(yDist &classDist, xxxyDist & dist, std::vector<CategoricalAttribute> &parentIsC_order, CategoricalAttribute x, std::vector<CategoricalAttribute> &temp, std::vector<std::vector<CategoricalAttribute> > &parents, std::vector<CategoricalAttribute> &order, const instance & inst);
//基于nb,k=2,找父节点过程
void findmink_k2(yDist &classDist, xxxyDist & dist, CategoricalAttribute x, std::vector<CategoricalAttribute> &temp, std::vector<std::vector<CategoricalAttribute> > &parents, std::vector<CategoricalAttribute> &porder, const instance & inst);

void H_recursion_del(yDist &classDist, xxxyDist & dist, double H_standard, unsigned int arc, std::vector<std::vector<CategoricalAttribute> > &parents_, std::vector<std::vector<CategoricalAttribute> > &parents_after, const instance & inst);
void displayInfo(yDist &classDist_, xxxyDist & dist_1, unsigned int noCatAtts_,unsigned int noClasses_,const instance& inst, std::vector<std::vector<CategoricalAttribute> > parents_);
void H_recursion_dp_k2(yDist &classDist, xxxyDist & dist, double H_standard, unsigned int arc, std::vector<CategoricalAttribute> &order,
        std::vector<std::vector<CategoricalAttribute> > &parents_,
        std::vector<std::vector<CategoricalAttribute> > &parents_after, const instance & inst);
bool search1(std::vector<CategoricalAttribute> qwe, int k);
bool search2(std::vector<CategoricalAttribute> ch, std::vector<CategoricalAttribute> fa, CategoricalAttribute a, CategoricalAttribute b);
void H_rec_p(yDist &classDist, xxxyDist & dist, double H_standard, unsigned int arc, std::vector<CategoricalAttribute> &order,
        std::vector<std::vector<CategoricalAttribute> > &parents_,
        std::vector<std::vector<CategoricalAttribute> > &parents_after, const instance & inst, std::vector<CategoricalAttribute> qwe, std::vector<CategoricalAttribute> ch, std::vector<CategoricalAttribute> fa);

bool depthParent(xxxyDist & dist,CategoricalAttribute node,CategoricalAttribute value,std::vector<std::vector<CategoricalAttribute> > pm);


double getIxxx(xxxyDist &dist, CategoricalAttribute xi, CategoricalAttribute xj, CategoricalAttribute xk, std::vector<CategoricalAttribute> parents_1);
void getMutualInformation(xyDist &dist, std::vector<float> &mi);
void getMutualInformationTC(xyDist &dist, crosstab<double> &mi);
void getMutualInformationTCloc(xyDist &dist, crosstab<float> &mi,const instance & inst);
void getMutualInformationloc(xyDist &dist, std::vector<float> &mi, const instance& inst);

//!!!!*******************************************************************************************************

void getUnionMI(xxyDist &dist, crosstab<float> &mi_xxy);//联合互信息
void getUnmi_loc(xxyDist &dist, crosstab<float> &mi_xxy, const instance & inst);//局部联合互信息

//!!!!*******************************************************************************************************

double getInfoGain_loc(xyDist& dist, CategoricalAttribute a, const instance & inst);
double getInformation_loc(xyDist& dist, CategoricalAttribute a, const instance & inst);
double getGainRatio_loc(xyDist& dist, CategoricalAttribute a, const instance & inst);
void getH(xxyDist &dist, crosstab<float> &cmi);

void getXMI(xxyDist &dist, std::vector<float> &xmi,const CategoricalAttribute & attno, const unsigned int & attvalno);
void getXMI_loc(xxyDist &dist, std::vector<float> &xmi, const CategoricalAttribute & attno, const unsigned int & attvalno, const instance & inst);

void getGeneralMutualInformation(xyDist &dist, std::vector<std::vector<float> > &gmi);
void getXCondMutualInf(xxyDist &dist, crosstab<float> &DoubleMI);

/**
 * Calculates the symmetrical uncertainty between the attributes in dist and the
 * class
 * 
 * SU(X;Y) = 2 . MI(X,C) / ( H(X)+H(C))
 * 
 * @param dist  counts for the xy distributions.
 * @param[out] su symmetrical uncertainty between the attributes and the class.
 */
void getSymmetricalUncert(xyDist &dist, std::vector<float> &su);


/**
 * Calculates the class conditional mutual information between the attributes in 
 * dist conditioned on the class
 * 
 * CMI(X;Y|C) = H(X|C) - H(X|Y,C)
 * 
 * @param dist  counts for the xy distributions.
 * @param[out] cmi class conditional mutual information between the attributes.
 */
void getCondMutualInf(xxyDist &dist, crosstab<float> &cmi);
void getCondMutualInfTC(xxyDist &dist,crosstab3D<double>&cmi);
void getCondMutualInfTCloc(xxyDist &dist, crosstab3D<float> &cmi, const instance & inst) ;
void getCondMutualInfloc(xxyDist &dist, crosstab<float> &cmi, const instance& inst);
void getCondMutualInflocloc(xxyDist &dist, CategoricalAttribute y, crosstab<float> &cmi, const instance& inst);


void getUnion3cmi(xxxyDist &dist, crosstab3D<float> &cmi);
//!!!!*******************************************************************************************************
void getUnionCmi(xxxyDist &dist, crosstab3D<float> &cmi);//联合条件互信息
void getUnionCmi_loc(xxxyDist &dist, crosstab3D<float> &cmi, const instance & inst);//局部联合条件互信息
//!!!!*******************************************************************************************************

void getCMI_Ratio(xxyDist &dist, crosstab<float> &cmi);
void getCMIxxy(xxyDist &dist, crosstab<float> &cmi_xxy, crosstab<float> &cmi_xxy_ratio);
void getCMIxxy2(xxyDist &dist, crosstab<float> &cmi_xxy, crosstab<float> &cmi_xxy_Ixx);
void getCMIxyx(xxyDist &dist, crosstab<float> &cmi_xyx, crosstab<float> &cmi_xyx_ratio);


void getXCMI(xxxyDist &dist, crosstab<float> &xcmi, const CategoricalAttribute & attno, const CategoricalAttribute & attvalno);
void getXCMI_loc(xxxyDist &dist, crosstab<float> &xcmi, const CategoricalAttribute & attno, const CategoricalAttribute & attvalno, const instance & inst);

/**
 * Calculate the difference between the probabilities considering independency
 * over dependency between the attributes X and Y: abs ( p(X|C) - P(X|Y,C) ).
 * 
 * @param dist  counts for the xy distributions.
 * @param[out] cm class conditional correlation measure between the attributes.
 */
void getDoubleMutualInf(xxyDist &dist, crosstab<float> &DoubleMI);

/**
 * Calculates the mutual information between the attributes in dist and the
 * class
 * 
 * DoubleMI(Xi;Xj,C) = H(Xi) - H(Xi|Xj,C)
 * 
 * @param dist  counts for the xy distributions.
 * @param[out] mi mutual information between the attributes and the class.
 */
void getErrorDiff(xxyDist &dist, crosstab<float> &cm);


/**
 * Calculates the class conditional symmetrical uncertainty between the 
 * attributes in dist conditioned on the class
 * 
 * CSU(X;Y|C) = 2 . CMI(X;Y|C)/( H(X|C) + H(Y|C) )
 * 
 * @param dist  counts for the xy distributions.
 * @param[out] csu class conditional symmetrical uncertainty between the attributes.
 */
void getCondSymmUncert(xxyDist &dist, crosstab<float> &csu);


/**
 * Chi-squared test for independence, one attribute against the class
 *  
 * @param cells counts for the attribute's values and the class 
 * @param rows number of values for the attribute
 * @param cols number of classes
 * 
 * @return The complemented Chi-square distribution
 */
double chiSquare(const InstanceCount *cells, const unsigned int rows,
        const unsigned int cols);



/**
 * Calculates the conditional mutual information between the class and the attributes in 
 * dist conditioned on another attribute in dist
 * 
 * ACMI(X;C|Y) = H(X|Y) - H(X|Y,C)
 * 
 * @param dist  counts for the xy distributions.
 * @param[out] atcmi conditional mutual information between the attributes and the class
 * given another attribute.
 */
void getAttClassCondMutualInf(xxyDist &dist, crosstab<float> &acmi);

/**
 * Calculates the class conditional mutual information between the attributes in 
 * dist conditioned on the class and another attribute
 * 
 * MCMI(X;Z|C,Y) = H(X|C,Y) - H(X|Z,C,Y)
 * 
 * @param dist  counts for the xy distributions.
 * @param[out] mcmi class conditional mutual information between the attributes in 
 * dist conditioned on the class and another attribute
 */
void getMultCondMutualInf(xxxyDist &dist, std::vector<crosstab<float> > &mcmi);


/**
 * Calculates the following measures at the same time:
 * CMI(X;Y|C) = H(X|C) - H(X|Y,C)
 * ACMI(X;C|Y) = H(X|Y) - H(X|Y,C)
 * 
 * @param dist  counts for the xy distributions.
 * @param[out] cmi class conditional mutual information between the attributes.
 * @param[out] acmi conditional mutual information between the attributes and the class
 * given another attribute.
 */
void getBothCondMutualInf(xxyDist &dist, crosstab<float> &cmi,
        crosstab<float> &acmi);

/// Calculates the Matthew's Correlation Coefficient from a set of TP, FP, TN, FN counts

inline double calcBinaryMCC(const InstanceCount TP, const InstanceCount FP, const InstanceCount TN, const InstanceCount FN) {
    if (TP + TN == 0) return -1.0;
    if (FP + FN == 0) return 1.0;
    if ((TP + FN == 0) || (TN + FP == 0) || (TP + FP == 0) || (TN + FN == 0)) return 0.0;
    else return (static_cast<double> (TP) * TN - static_cast<double> (FP) * FN)
        / sqrt(static_cast<double> (TP + FP)
            * static_cast<double> (TP + FN)
            * static_cast<double> (TN + FP)
            * static_cast<double> (TN + FN));

}

/**
 * Calculates the Matthew's Correlation Coefficient given a confusion matrix (any number of classes) as in http://rk.kvl.dk/
 * 
 * @param xtab  confusion matrix.
 * @return the MCC
 */

double calcMCC(crosstab<InstanceCount> &xtab);


/**
 * Calculate the symmetrical uncertainty between two attributes
 */
double getSymmetricalUncert(const xxyDist &dist, CategoricalAttribute x1, CategoricalAttribute x2);

