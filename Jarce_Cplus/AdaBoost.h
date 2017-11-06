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
 **
 **
 */

#pragma once

#include "learner.h"
#include "StoredInstanceStream.h"
#include "StoredIndirectInstanceStream.h"

#include "instanceSample.h"


class AdaBoost: public learner {
public:
	AdaBoost(char* const *& argv, char* const * end);
	virtual ~AdaBoost();

	void getCapabilities(capabilities &c);

	/**
	 * trains the adaboost committee of learners.
	 *
	 * @param is The training set
	 */
	virtual void train(InstanceStream &is);

	/**
	 * Calculates the class membership probabilities for the given test instance.
	 *
	 * @param inst The instance to be classified
	 * @param classDist Predicted class probability distribution
	 */
	virtual void classify(const instance &inst, std::vector<double> &classDist);

private:


	/**
	 * Select only instances with weights that contribute to
	 * the specified quantile of the weight distribution
	 *
	 * @param thisStream Selected instances
	 * @param aStream  The instances source
	 * @param quantile  the specified quantile eg 0.9 to select
     * 90% of the weight mass
	 */
	void selectWeightQuantile(StoredIndirectInstanceStream &thisStream,std::vector<double> &weight,
			AddressableInstanceStream* aStream, double quantile);

	/**
	   * Creates a new dataset of the same size using random sampling
	   * with replacement according to the given weight vector.
	   * 	   *
	   * @sampleStream the returned dataset
	   * @sourceStream  the source dataset
	   * @param weight the weight vector
	   * @param rand a random number generator
	   */

	void resampleWithWeights(StoredIndirectInstanceStream &sampleStream,
			StoredIndirectInstanceStream &sourceStream, const std::vector<float> &weight, MTRand &rand);


	void sampleAndUpdateWeights(InstanceSample &sampleStream,
			InstanceStream &sourceStream,double epsilon);

	char const* learnerName_;           ///< the name of the learner
	char* const * learnerArgv_;  ///< the start of the arguments for the learner
	char* const * learnerArgEnd_;   ///< the end of the arguments ot the learner
	std::vector<learner*> classifiers_; ///< the classifiers in the ensemble
	unsigned int size_;         ///< the number of boosted classifiers to create
	unsigned int weightThreshold_; ///< the threshold for selecting how many instances
	std::vector<float> weight_;			///< weight for each instance
	std::vector<double>   Betas_;     ///<
	InstanceCount sampleSize_;   	///< size of the sampled instance set
	InstanceCount dataSize_; 		///< size of the source instance stream
	bool firstScan_;  				///< if it is the first scan of the data
	std::vector<bool> predicted_;
	capabilities capabilities_;
	  MTRand randSampleInstance_;
	  MTRand randReplaceInstance_;
};

