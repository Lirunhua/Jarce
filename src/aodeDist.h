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

#pragma once

#include "incrementalLearner.h"
#include "smoothing.h"


class aodeDist: public IncrementalLearner {
public:
	/**
	 * @param argv Options for the aode classifier
	 * @param argc Number of options for aode
	 * @param m    Metadata information
	 */
	aodeDist(char* const *& argv, char* const * end);

	virtual ~aodeDist(void);

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
	void nbClassify(const instance &inst, std::vector<double> &classDist);
	/**
	 * Type of subsumption resolution type.
	 */
//	typedef enum {
//		srtNone, /**< no subsumption resolution.*/
//		srtBSE, /**< backwards sequential elimination. */
//		srtLSR, /**< lazy subsumption resolution. */
//		srtESR, /**< eager subsumption resolution. */
//		srtNSR, /**< near-subsumption resolution. */
//	} subsumptionResolutioniType;
//
//	subsumptionResolutioniType srt;

//	 p(a=v, Y=y) using M-estimate
	inline double jointP(CategoricalAttribute a, CatValue v, CatValue y) {
		return (xyCounts_[attOffset[a] + v][y]
				+ M / (instanceStream_->getNoValues(a) * noClasses_))
				/ (count + M);
	}

	// p(x1=v1, x2=v2, Y=y) using M-estimate
	inline double jointP(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2, CatValue y) const {

		InstanceCount countxxy;
		if (x1 > x2)
			countxxy = xxyCounts_[x1][v1][x2][v2 ][ y];
		else
			countxxy = xxyCounts_[x2][v2][x1][v1][ y];

		return (countxxy
				+ M
						/ (instanceStream_->getNoValues(x1)
								* instanceStream_->getNoValues(x2) * noClasses_))
				/ (count + M);
	}

	// p(x1=v1, x2=v2) using M-estimate
	inline double jointP(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2) const {
		return (getCount(x1, v1, x2, v2)
				+ M
						/ (instanceStream_->getNoValues(x1)
								* instanceStream_->getNoValues(x2)))
				/ (count + M);
	}

	// p(Y=y) using M-estimate
	inline double p(CatValue y) {
		return (classCounts[y] + M / noClasses_) / (count + M);
	}

	// p(a=v|Y=y) using M-estimate
	inline double p(CategoricalAttribute a, CatValue v, CatValue y) {
		return mEstimate(xyCounts_[attOffset[a] + v][y], classCounts[y],
				instanceStream_->getNoValues(a));
	}

	// p(x1=v1|Y=y, x2=v2) using M-estimate
	inline double p(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2, CatValue y) const {
		InstanceCount countxxy;
		if (x1 > x2)
			countxxy = xxyCounts_[x1][v1][x2][v2 ][ y];
		else
			countxxy = xxyCounts_[x2][v2][x1][v1 ][ y];

		InstanceCount countxy = xyCounts_[attOffset[x2] + v2][y];

		return (countxxy + M / instanceStream_->getNoValues(x1)) / (countxy + M);
	}

	// count for instances x1=v1, x2=v2
	inline InstanceCount getCount(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2) const {
		InstanceCount c = 0;

		for (CatValue y = 0; y < noClasses_; y++) {
			if (x1 > x2)
				c += xxyCounts_[x1][v1][x2][v2 ][ y];
			else
				c += xxyCounts_[x2][v2][x1][v1 ][ y];
		}
		return c;
	}

	// count for instances x1=v1
	inline InstanceCount getCount(CategoricalAttribute x, CatValue v) const {
		InstanceCount c = 0;

		for (CatValue y = 0; y < noClasses_; y++) {
			c += xyCounts_[attOffset[x] + v][y];
		}
		return c;
	}

	InstanceStream* instanceStream_;

	std::vector<int> generalizationSet;
	std::vector<int> substitutionSet;

	unsigned int minCount;

	bool subsumptionResolution;

	bool weighted; 			///< true  if  using weighted aode
	std::vector<float> weight; ///<stores the mutual information between each attribute and class as weight

	unsigned int noCatAtts_;  ///< the number of categorical attributes.
	unsigned int noClasses_;  ///< the number of classes
	bool trainingIsFinished_; ///< true iff the learner is trained

	//xxyDistSl xxyDist_; ///< the xxy distribution that aode learns from the instance stream and uses for classification

	std::vector<int> attOffset;

	InstanceCount count;
	std::vector<InstanceCount> classCounts;
	std::vector<std::vector<InstanceCount> > xyCounts_;
	std::vector<std::vector<std::vector<std::vector<std::vector<InstanceCount> > > > > xxyCounts_;

	std::vector<std::vector<std::vector<std::vector<std::vector<double> > > > > condiProbs;

};

