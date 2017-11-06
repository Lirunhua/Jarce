/*
 * a3de.h
 *
 *  Created on: 28/09/2012
 *      Author: shengleichen
 */

#pragma once

#include "incrementalLearner.h"
#include "xxxxyDist.h"
#include "crosstab.h"
#include "crosstab3d.h"
class a3de: public IncrementalLearner {
public:

	a3de(char* const *& argv, char* const * end);

	virtual ~a3de();

	void reset(InstanceStream &is);   ///< reset the learner prior to training

	bool trainingIsFinished(); ///< true iff no more passes are required. updated by finalisePass()

	/**
	 * Inisialises the pass indicated by the parametre.
	 *
	 * @param pass  Current pass.
	 */
	void initialisePass();
	/**
	 * Train an a3de with instance inst.
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
	 * Calculates the weight for wa3de
	 */
	void finalisePass();

	void getCapabilities(capabilities &c);

private:

	unsigned int minCount;

	bool subsumptionResolution; ///< true if using lazy subsumption resolution
	bool weighted; 			///< true  if  using weighted aode
	bool su_;
	bool mi_;
	bool chisq_;
	bool acmi_;

	float factor_;
	bool avg_;
	bool sum_;

	bool oneSelective_;
	bool twoSelective_;

	InstanceStream* instanceStream_;

	crosstab<double> weight_a2de; ///<stores the mutual information between each attribute and class as weight

	std::vector<double> weight_aode; ///<stores the mutual information between each attribute and class as weight

	crosstab3D<double> weight; ///<stores the mutual information between each attribute and class as weight


//	std::vector<int> generalizationSet;
//	std::vector<int> substitutionSet;

	std::vector<bool> generalizationSet;

	std::vector<bool> active_; ///< true for active[att] if att is selected -- flags: chisq, selective, selectiveTest
	unsigned int inactiveCnt_; ///< number of attributes not selected -- flags: chisq, selective, selectiveTest


	//std::vector<CategoricalAttribute> selectedAtt;
	bool memorySelective_;
	bool selected;
	bool trainingIsFinished_; ///< true iff the learner is trained
	unsigned int noCatAtts_;  ///< the number of categorical attributes.
	unsigned int noClasses_;  ///< the number of classes
//    bool printSelective_; ///<true if printing how many attributes have been selected
	xxxxyDist xxxxyDist_;

	void nbClassify(const instance &inst, std::vector<double> &classDist,
			xyDist &dist);
	void aodeClassify(const instance &inst, std::vector<double> &classDist,
			xxyDist & dist);
	void a2deClassify(const instance &inst, std::vector<double> &classDist,
			xxxyDist & dist);

};

