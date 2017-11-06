/*
 * a2de3.h
 *
 *  Created on: 28/09/2012
 *      Author: shengleichen
 */

#pragma once

#include "incrementalLearner.h"
#include "xxxyDist3.h"
#include "crosstab.h"
class a2de3: public IncrementalLearner {
public:

	a2de3(char* const *& argv, char* const * end);

	virtual ~a2de3();

	void reset(InstanceStream &is);   ///< reset the learner prior to training

	bool trainingIsFinished(); ///< true iff no more passes are required. updated by finalisePass()

	/**
	 * Inisialises the pass indicated by the parametre.
	 *
	 * @param pass  Current pass.
	 */
	void initialisePass();
	/**
	 * Train an aode with instance inst.
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
	 * Calculates the weight for wa2de3
	 */
	void finalisePass();

	void getCapabilities(capabilities &c);

private:

	//unsigned int minCount;

	//bool subsumptionResolution; ///< true if using lazy subsumption resolution
	//bool weighted; 			///< true  if  using weighted a2de3
//bool avg;  				///< true if using averaged mutual information as weight

	bool selected;   			///< true iif using selective a2de3

	bool su_;			///<true if using symmetric uncertainty to select active attributes
	bool mi_;			///<true if using mutual information to select active attributes
	bool chisq_;		///< true if using chi square test to selects active attributes
	bool acmi_;

	bool avg_;
	bool sum_;
    bool empiricalMEst_;  ///< true if using empirical m-estimation
    bool empiricalMEst2_;  ///< true if using empirical m-estimation of attribute given parent


    bool memorySelective_; ///<true if determining the number of selected attributes according to the space required in aode
	bool trainingIsFinished_; ///< true iff the learner is trained

	InstanceStream* instanceStream_;
	crosstab<double> weight; ///<stores the mutual information between each attribute and class as weight

	std::vector<double> weightaode; ///<stores the mutual information between each attribute and class as weight

//	std::vector<int> generalizationSet;
//	std::vector<int> substitutionSet;

	std::vector<bool> generalizationSet; 		///< indicate if att should be delete according to subsumption resolution

	std::vector<bool> active_; ///< true for active[att] if att is selected -- flags: chisq, selective, selectiveTest
	unsigned int inactiveCnt_; ///< number of attributes not selected -- flags: chisq, selective, selectiveTest
	float factor_;      ///< number of  mutual information selected attributes is the original number by factor_

	//std::vector<CategoricalAttribute> selectedAtt;

	std::vector<CategoricalAttribute> order_;
	unsigned int pass_;

	unsigned int noSelectedCatAtts_; ///< the number of categorical CategoricalAttributes.
	unsigned int noUnSelectedCatAtts_; ///< the number of categorical CategoricalAttributes.

	unsigned int noCatAtts_;  ///< the number of categorical attributes.
	unsigned int noClasses_;  ///< the number of classes

	InstanceCount count;
	xxxyDist3 xxxyDist_;
	xxyDist xxyDist_;

	void nbClassify(const instance &inst, std::vector<double> &classDist,
			xyDist &dist);
	void aodeClassify(const instance &inst, std::vector<double> &classDist,
			xxyDist & dist);
};

