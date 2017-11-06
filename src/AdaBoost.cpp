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

#include "AdaBoost.h"
#include "learnerRegistry.h"
#include "utils.h"

#include "StoredInstanceStream.h"
#include "StoredIndirectInstanceStream.h"
#include "mtrand.h"
#include "globals.h"

AdaBoost::AdaBoost(char* const *& argv, char* const * end) {
	// TODO Auto-generated constructor stub
	learner *theLearner = NULL;
	name_ = "AdaBoost";
	weightThreshold_ = 100;
	sampleSize_ = 10000;
	size_ = 10;
	firstScan_ = true;

	// get arguments
	while (argv != end) {
		if (*argv[0] != '-') {
			break;
		} else if (argv[0][1] == 'b') {
			// specify the base learner
			learnerName_ = argv[0] + 2;
			learnerArgv_ = ++argv;

			// create the learner
			theLearner = createLearner(learnerName_, argv, end);

			if (theLearner == NULL) {
				error("Learner %s is not supported", learnerName_);
			}

			learnerArgEnd_ = argv;

			name_ += "_";
			name_ += *theLearner->getName();
			break;
		} else if (argv[0][1] == 't') {
			getUIntFromStr(argv[0] + 2, weightThreshold_, "t");
			if (weightThreshold_ > 100)
				error("The weight threshold can not be greater than 100.\n");
		} else if (argv[0][1] == 'm') {
			getUIntFromStr(argv[0] + 2, sampleSize_, "m");
			name_ += argv[0];
		} else if (argv[0][1] == 's') {
			getUIntFromStr(argv[0] + 2, size_, "s");
			name_ += argv[0];
		} else {
			error("AdaBoost does not support argument %s\n", argv[0]);
			break;
		}

		++argv;
	}

	if (theLearner == NULL)
		error("No base learner specified");
	else {
		theLearner->getCapabilities(capabilities_);
		delete theLearner;
	}

	printf("Classifier %s is constructed.\n", name_.c_str());
	printf("The size of sampled data sets is: %u\n", sampleSize_);
}

AdaBoost::~AdaBoost() {
	// TODO Auto-generated destructor stub

	for (unsigned int i = 0; i < classifiers_.size(); ++i) {
		delete classifiers_[i];
	}
}
void AdaBoost::getCapabilities(capabilities &c) {
	c = capabilities_;
}

// creates a comparator for two double value
class valCmpClass {
public:
	valCmpClass(std::vector<double> *s) {
		val = s;
	}

	bool operator()(unsigned int a, unsigned int b) {
		return (*val)[a] > (*val)[b];
	}

private:
	std::vector<double> *val;
};

//Select only instances with weights that contribute to
//the specified quantile of the weight distribution
void AdaBoost::selectWeightQuantile(StoredIndirectInstanceStream &thisStream,
		std::vector<double> &weight, AddressableInstanceStream* aStream,
		double quantile) {
//	const InstanceCount dataSize = aStream->size();
//	//std::vector<double> weight(weight_); ///< store the weights for each instance
//
//	thisStream.setSourceWithoutLoading(*aStream);  // clear the stream
//	double sumOfWeights = 0;
//
//	std::vector<unsigned int> order;
//	for (unsigned int a = 0; a < dataSize; a++) {
//		order.push_back(a);
//	}
//
//	if (!order.empty()) {
//		valCmpClass cmp(&weight);
//		std::sort(order.begin(), order.end(), cmp);
//	}
//
//	for (unsigned int a = 0; a < dataSize; a++) {
//		sumOfWeights += weight[order[a]];
//		aStream->goTo(order[a]+1);
//		thisStream.add(aStream->current());
//
//		if ((sumOfWeights > quantile) && (a < dataSize - 1)
//				&& (weight[order[a]] != weight[order[a + 1]]))
//			break;
//	}
//
//	// remove the additional elements in weight
//	// so as to remain the length of thisStream as the length of weight
//	for (unsigned int a = dataSize; a > thisStream.size(); a--)
//		weight.pop_back();
//
//	assert(weight.size()==thisStream.size());
}
void AdaBoost::resampleWithWeights(StoredIndirectInstanceStream &sampleStream,
		StoredIndirectInstanceStream &sourceStream,
		const std::vector<float> &weight, MTRand &rand) {

	sourceStream.rewind();
	sampleStream.setSourceWithoutLoading(sourceStream);

	assert(weight.size()==sourceStream.size());

	const InstanceCount dataSize = weight.size();
	std::vector<double> probabilities(dataSize);
	double sumProbs = 0;
	double sumOfWeights = sum(weight);

	//calculate the cumulative probability function
	for (unsigned int i = 0; i < dataSize; i++) {
		sumProbs += rand();
		probabilities[i] = sumProbs;
	}

	//normalise only the selected weight
	for (unsigned int i = 0; i < dataSize; i++) {
		probabilities[i] = probabilities[i] * sumOfWeights / sumProbs;
	}

	// Make sure that rounding errors don't mess things up
	probabilities[dataSize - 1] = sumOfWeights;

	unsigned int k = 0;
	unsigned int l = 0;
	sumProbs = 0;
	while ((k < dataSize && (l < dataSize))) {
		if (weight[l] < 0) {
			error("Weights have to be positive.");
		}
		sumProbs += weight[l];
		while ((k < dataSize) && (probabilities[k] <= sumProbs)) {
			sourceStream.goTo(l + 1);
			sampleStream.add(sourceStream.current());
			k++;
		}
		l++;
	}

}
void AdaBoost::sampleAndUpdateWeights(InstanceSample &sampleStream,
		InstanceStream &sourceStream, double epsilon) {

	InstanceCount count = 0;
	instance inst(sourceStream);
	sourceStream.rewind();

	//scan the instance stream only once
	//during which update the weight and sample instances
	while (!sourceStream.isAtEnd()) {
		if (sourceStream.advance(inst)) {

			if (firstScan_ == true)
				weight_.push_back(1);
			else {

				if (epsilon < 0.5 && epsilon > 0) {
					if (verbosity >= 3) {

						printf("Output the weight for instance %u:\n", count);
					}
					if (predicted_[count] == true) {
						if (verbosity >= 3) {
							printf("predicted true, weight before: %f\n",
									weight_[count]);
						}
						weight_[count] /= (2 * (1 - epsilon));
						if (verbosity >= 3) {
							printf("predicted true, weight after: %f\n",
									weight_[count]);
						}

					} else {
						if (verbosity >= 3) {
							printf("predicted false, weight before: %f\n",
									weight_[count]);
						}
						weight_[count] /= (2 * epsilon);
						if (verbosity >= 3) {
							printf("predicted false, weight after: %f\n",
									weight_[count]);
						}
					}
				}
			}
			sampleStream.sampleWithWeights(inst, weight_,randSampleInstance_,randReplaceInstance_);
			count++;
		}
	}

	if (verbosity >= 2) {
		printf("The sum of weight for all instances: %f\n", sum(weight_));
	}
	if (firstScan_ == true) {
		dataSize_ = weight_.size();
		firstScan_ = false;
	}

}

void AdaBoost::train(InstanceStream &is) {

	// load the data into a store
	//StoredInstanceStream store;
	//IndirectInstanceStream thisStream; ///< the stream for learning the next classifier
	//AddressableInstanceStream* aStream =
	//		dynamic_cast<AddressableInstanceStream*>(&is);
	//MTRand rand;

	// if 'is' is not an instance of AddressableInstancesStream,
	// store instances in memory
//	if (aStream == NULL) {
//		store.setSource(is);
//		aStream = &store;
//	}

	// reset the classifier
	for (unsigned int i = 0; i < classifiers_.size(); ++i)
		delete classifiers_[i];
	classifiers_.clear();

	//initialise the weight for training instances
	//const InstanceCount dataSize = aStream->size();
	//weight_.assign(dataSize, 1.0 / dataSize);

	unsigned int invalidCount = 0;
	double epsilon = 0;
	for (unsigned int i = 0, j = 0; i < size_; ++j) {

//		// Select instances to train the classifier according to specification
//		if (weightThreshold_ ==100) {
//
//			selectWeightQuantile(thisStream, weight, aStream,
//					weightThreshold_ / 100.0);
//
//		} else {
//			//select all the instances
//			thisStream.setSource(*aStream);
//		}
		if (verbosity >= 3) {
			printf("\nStart of iteration: %u\n------------------------\n", j);
		}
		// Resample
		InstanceSample sampleStream(sampleSize_); ///< the resampled stream for learning the next classifier

		sampleStream.setSource(is);


//		aStream->rewind();
//		thisStream.setSource(*aStream);
		//resampleWithWeights(sampleStream, thisStream, weight_,rand);

		sampleAndUpdateWeights(sampleStream, is,epsilon);

		//create the classifier and train
		classifiers_.push_back(
				createLearner(learnerName_, learnerArgv_, learnerArgEnd_));
		classifiers_.back()->train(sampleStream);

		//evaluate the classifier on whole data sets
		InstanceCount correctCount = 0;
		InstanceCount count = 0;
		std::vector<double> classDist(is.getNoClasses());
		instance inst(is);

		//double restWeight=0;
		epsilon = 0;
		is.rewind();
		predicted_.clear();
		while (!is.isAtEnd()) {
			if (is.advance(inst)) {

				classifiers_.back()->classify(inst, classDist);

				const CatValue prediction = indexOfMaxVal(classDist);
				const CatValue trueClass = inst.getClass();

				if (prediction != trueClass) {
					predicted_.push_back(false);
					epsilon += weight_[count];

				} else {
					predicted_.push_back(true);
					correctCount++;
					//restWeight+=weight_[count];
				}
				count++;
			}
		}

		epsilon /= dataSize_;
		//restWeight /= dataSize_;

		//assert(count==dataSize);
		printf(
				"For classifier %u, %u instances have been correctly classified, epsilon : %f.\n",
				i, correctCount, epsilon);

		if (epsilon == 0) {
			Betas_.push_back(0.0000000001);
			i++;
			printf("epsilon %20.20f is too small.\n", epsilon);
		} else if (fabs(epsilon-0.5)<0.0000001|| epsilon > 0.5) {
			//this can set the weight to 1 again
			firstScan_ = true;
			weight_.clear();
			//remove current classifier
			delete classifiers_.back();
			classifiers_.pop_back();
			invalidCount++;
			if (invalidCount > 10) {
				printf("Invalid count is greater than 10. exit. \n");
				break;
			}
			printf("epsilon:%f\n", epsilon);
		} else {
			// Determine the weight to assign to this model
			Betas_.push_back((1 - epsilon) / epsilon);
			i++;
			printf("beta: %f,epsilon:%f\n", Betas_.back(), epsilon);
		}

		if (verbosity >= 2) {
			printf("size of beta: %u"
					"\n", Betas_.size());

		}
		//when epsilon is zero, store beta but not update the weights

	}
	printf("the number of classifiers is :%u\n", Betas_.size());
	//assert(Betas_.size()==size_);
}

void AdaBoost::classify(const instance &inst, std::vector<double> &classDist) {

	std::vector<double> thisClassDist(classDist.size());

	classDist.assign(classDist.size(), 0.0);

	for (unsigned int i = 0; i < classifiers_.size(); ++i) {
		classifiers_[i]->classify(inst, thisClassDist);
		classDist[indexOfMaxVal(thisClassDist)] += log(1 / Betas_[i]);
	}

	normalise(classDist);
}

