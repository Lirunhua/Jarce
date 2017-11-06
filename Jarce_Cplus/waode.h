/* 
 * File:   waode.h
 * Author: dragonriveryu
 *
 * Created on 2014年9月14日, 上午9:59
 */

/*
 * 这个算法是在aode算法上利用一种加权平均的思想，
 * 产生新的算法WAODE
 */
#ifndef WAODE_H
#define	WAODE_H

#include "incrementalLearner.h"
#include "xxxxyDist.h"
#include "xxxyDist.h"

class waode: public IncrementalLearner {
public:
	/**
	 * @param argv Options for the hnb_aode classifier
	 * @param argc Number of options for hnb_aode
	 * @param m    Metadata information
	 */
	waode(char* const *& argv, char* const * end);

	virtual ~waode(void);

	void reset(InstanceStream &is);   ///< reset the learner prior to training

	bool trainingIsFinished(); ///< true iff no more passes are required. updated by finalisePass()

	/**
	 * Inisialises the pass indicated by the parametre.
	 *
	 * @param pass  Current pass.
	 */
	void initialisePass();
	/**
	 * Train an hnb_ao with instance inst.
	 *
	 * @param inst Training instance
	 */
	void train(const instance &inst);

	/**
	 * Calculates the class membership probabilities for the given test instance.
	 *
	 * @param inst The instance to be classified
	 * @param classDist Predicted class probability distribution
	 */
	void classify(const instance &inst, std::vector<double> &classDist);
	/**
	 * Calculates the weight for waode
	 */
	void finalisePass();

	void getCapabilities(capabilities &c);

private:
	/**
	 * Naive Bayes classifier to which aode will deteriorate when there are no eligible parent attribute (also as SPODE)
	 *
	 *@param inst The instance to be classified
	 *@param classDist Predicted class probability distribution
	 *@param dist  class object pointer of xyDist describing the distribution of attribute and class
	 */
	void nbClassify(const instance &inst, std::vector<double> &classDist,
			xyDist &xyDist_);

	int getNextElement(std::vector<CategoricalAttribute> &order,CategoricalAttribute ca, unsigned int noSelected);

//	std::vector<CategoricalAttribute> selectedAtt;
	unsigned int minCount;
        float UsedAttrRatio;
	bool useAttribSelec_;
	unsigned int attribSelected_;
	bool subsumptionResolution; ///<true if selecting active parents for each instance
	bool su_;
	bool mi_;
	bool ig_;
	bool acmi_;
	bool chisq_;
        bool empiricalMEst_;  ///< true if using empirical m-estimation
        bool empiricalMEst2_;  ///< true if using empirical m-estimation of attribute given parent
	bool selected;
	bool correlationFilter_;
	bool useThreshold_;
	double threshold_;
	bool weighted; 			///< true  if  using weighted aode
	bool chilect_;       ///< true if selecting child
	std::vector<bool> chiactive_;  ///< true for chiactive[att] if att is selected as child
        unsigned int fathercount[100]; 

	std::vector<bool> active_; ///< true for active[att] if att is selected -- flags: chisq, selective, selectiveTest

	std::vector<double> weight; ///<stores the mutual information between each attribute and class as weight

	unsigned int inactiveCnt_; ///< number of attributes not selected -- flags: chisq, selective, selectiveTest


	float factor_;      ///< the number of  mutual information selected attributes  is the original number multiplied by factor_


	InstanceStream* instanceStream_;

	unsigned int noCatAtts_;  ///< the number of categorical attributes.
	unsigned int noClasses_;  ///< the number of classes
	bool trainingIsFinished_; ///< true iff the learner is trained
	xxyDist xxyDist_; ///< the xxy distribution that aode learns from the instance stream and uses for classification
        
        xyDist xyDist_;
        std::vector<float> wi;//保存的是属性变量和类变量之间的权值
        float sumOfwi;
};


#endif	/* WAODE_H */

