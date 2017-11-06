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
#pragma once

#include "incrementalLearner.h"
#include "xxxyDist.h"
#include <limits>

class vfan: public IncrementalLearner {
public:
	vfan();
	vfan(char* const *& argv, char* const * end);
	~vfan(void);

	void reset(InstanceStream &is);   ///< reset the learner prior to training
	void initialisePass(); ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
	void train(const instance &inst); ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
	void finalisePass(); ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
	bool trainingIsFinished(); ///< true iff no more passes are required. updated by finalisePass()
	void getCapabilities(capabilities &c);

	virtual void classify(const instance &inst, std::vector<double> &classDist);

private:
	unsigned int minCount;

	bool subsumptionResolution; ///< true if using lazy subsumption resolution
	bool weighted; 			///< true  if  using weighted a2de
	bool avg;  				///< true if using averaged mutual information as weight

	bool selected;   			///< true iif using selective a2de

	bool su_;			///<true if using symmetric uncertainty to select active attributes
	bool mi_;			///<true if using mutual information to select active attributes
	bool chisq_;		///< true if using chi square test to selects active attributes

        bool empiricalMEst_;  ///< true if using empirical m-estimation
        bool empiricalMEst2_;  ///< true if using empirical m-estimation of attribute given parent

	bool trainingIsFinished_; ///< true iff the learner is trained

	InstanceStream* instanceStream_;
	

	std::vector<double> weightaode; ///<stores the mutual information between each attribute and class as weight

//	std::vector<int> generalizationSet;
//	std::vector<int> substitutionSet;

	std::vector<bool> generalizationSet; 		///< indicate if att should be delete according to subsumption resolution

	std::vector<bool> active_; ///< true for active[att] if att is selected -- flags: chisq, selective, selectiveTest
	unsigned int inactiveCnt_; ///< number of attributes not selected -- flags: chisq, selective, selectiveTest
	unsigned int factor_;      ///< number of  mutual information selected attributes is the original number by factor_

	//std::vector<CategoricalAttribute> selectedAtt;


	unsigned int noCatAtts_;  ///< the number of categorical attributes.
	unsigned int noClasses_;  ///< the number of classes
	InstanceCount count;
        
	std::vector<CategoricalAttribute> parents_;
	xxxyDist xxxyDist_;
        int order[1000];
	
       
	const static CategoricalAttribute NOPARENT = 0xFFFFFFFFUL; // cannot use std::numeric_limits<categoricalAttribute>::max() because some compilers will not allow it here
};
