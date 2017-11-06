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
#include "biasvariance.h"
#include "biasVarianceInstanceStream.h"
#include "instance.h"
#include "utils.h"
#include "mtrand.h"
#include "globals.h"
#include <vector>
#include "crosstab.h"
#include "instanceStreamDiscretiser.h"

#ifdef __linux__
#include <sys/time.h>
#include <sys/resource.h>
#endif
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
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//void biasVariance(learner *theLearner, InstanceStream &instStream,
//	InstanceFile &instFile, FilterSet &filters, char* args) {

void biasVariance(learner *theLearner, InstanceStream &instStream,
		FilterSet &filters, char* args) {

	unsigned int noTrainingCases = 1000;
	unsigned int noTestCases = 1000;
	unsigned int noExperiments = 10;
	std::vector<unsigned int*> vals;

	vals.push_back(&noTrainingCases);
	vals.push_back(&noTestCases);
	vals.push_back(&noExperiments);

	getUIntListFromStr(args, vals, "bias variance settings");

	if (verbosity >= 1)
		printf("Counting size of %s\n", instStream.getName());

	const InstanceCount dataCount = instStream.size();

	if (dataCount < noTrainingCases + noTestCases) {
		error(
				"Too few cases for an experiment with %" ICFMT " training and %" ICFMT " test cases",
				noTrainingCases, noTestCases);
	}

	const unsigned int noClasses = instStream.getNoClasses();

	crosstab<InstanceCount> xtab(noClasses);

	unsigned int **results;
	safeAlloc(results, noTestCases);

	for (unsigned int i = 0; i < noTestCases; i++) {
		allocAndClear(results[i], noClasses);
	}

	instance inst(instStream);

	std::vector<double> classDist(noClasses);
	std::vector<double> zOLoss;  // 0-1 loss from each experiment
	std::vector<double> rmse;    // rmse from each experiment
	std::vector<double> rmsea;    // rmse for all classes from each experiment
	std::vector<double> logloss; // logarithmic loss for all classes from each experiment

	for (unsigned int exp = 0; exp < noExperiments; exp++) {
		if (verbosity >= 1)
			printf("Bias/variance experiment %d for %s\n", exp + 1,
					instStream.getName());

		InstanceCount count = 0;
		unsigned int zeroOneLoss = 0;
		double squaredError = 0.0;
		double squaredErrorAll = 0.0;
		double logLoss = 0.0;

		instStream.rewind();
		biasVarianceInstanceStream bvStream(&instStream, noTrainingCases,
				noTestCases, exp);

		bvStream.setTraining(true);
		InstanceStream* instanceStream = filters.apply(&bvStream);

		long int trainTime = 0;
		long int testTime = 0;

#ifdef __linux__
		struct rusage usage;
#endif

#ifdef __linux__
		getrusage(RUSAGE_SELF, &usage);
		trainTime = usage.ru_utime.tv_sec + usage.ru_stime.tv_sec;
#endif
//
//		{
//		unsigned int ji=0;
//
//			while (instanceStream->advance(inst)) {
//
//				if (verbosity >= 2) {
//					ji++;
//					printf("%u:##",ji);
//					for (unsigned int i = 0; i <14; i++)
//						printf("%d,", inst.getCatVal(i));
//					printf("\ttrain set:\n");
//				}
//			}
//
//			instanceStream->rewind();
//
//		}

		theLearner->train(*instanceStream);


#ifdef __linux__
		getrusage(RUSAGE_SELF, &usage);
		trainTime =
				((usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) - trainTime);
#endif

		bvStream.setTraining(false);

		instanceStream->rewind();

		instance inst(*instanceStream); // create a test instance

#ifdef __linux__
		getrusage(RUSAGE_SELF, &usage);
		testTime = usage.ru_utime.tv_sec + usage.ru_stime.tv_sec;
#endif
//
//		unsigned int ji=0;
//		printf("test set:\n");
//			while (instanceStream->advance(inst)) {
//
//				if (verbosity >= 2) {
//					ji++;
//					printf("%u:##",ji);
//					for (unsigned int i = 0; i <14; i++)
//						printf("%d,", inst.getCatVal(i));
//					printf("\ttest set \n");
//				}
//			}
//
//			instanceStream->rewind();
//


		if (!strncmp(theLearner->getName()->c_str(), "KDB-CondDisc", 12)) {
			while (!instanceStream->isAtEnd() && count < noTestCases) {
				if (static_cast<InstanceStreamDiscretiser*>(instanceStream)->advanceNumeric(
						inst)) {
					count++;

					theLearner->classify(inst, classDist);

					const CatValue prediction = indexOfMaxVal(classDist);
					const CatValue trueClass = inst.getClass();

					if (prediction != trueClass) {
						zeroOneLoss++;
					}

					results[count - 1][prediction]++;

					const double error = 1.0 - classDist[trueClass];

					squaredError += error * error;
					squaredErrorAll += error * error;
					logLoss += log2(classDist[trueClass]);
//					foldsquaredError += error * error;
//					foldsquaredErrorAll += error * error;
//					foldlogLoss += log2(classDist[trueClass]);
					for (CatValue y = 0; y < instanceStream->getNoClasses();
							y++) {
						if (y != trueClass) {
							const double err = classDist[y];
							squaredErrorAll += err * err;
//							foldsquaredErrorAll += err * err;
						}
					}

					xtab[trueClass][prediction]++;
				}
			}
		} else {
			while (!instanceStream->isAtEnd() && count < noTestCases) {
				if (instanceStream->advance(inst)) {
					count++;

					theLearner->classify(inst, classDist);
//
//
//			          if(verbosity>=2)
//			          {
//			        	   if(count==1)
//			        	   {
//			        		   printf("dist of test instance %d\n",count);
//			        		   print(classDist);
//			        		   printf("\n");
//
//			        	   }
//			          }
					const CatValue prediction = indexOfMaxVal(classDist);
					const CatValue trueClass = inst.getClass();

					if (prediction != trueClass) {
						zeroOneLoss++;
						//	foldzeroOneLoss++;
					}
					results[count - 1][prediction]++;

					const double error = 1.0 - classDist[trueClass];

					squaredError += error * error;
					squaredErrorAll += error * error;
					logLoss += log2(classDist[trueClass]);
//					foldsquaredError += error * error;
//					foldsquaredErrorAll += error * error;
//					foldlogLoss += log2(classDist[trueClass]);
					for (CatValue y = 0; y < instanceStream->getNoClasses();
							y++) {
						if (y != trueClass) {
							const double err = classDist[y];
							squaredErrorAll += err * err;
//							foldsquaredErrorAll += err * err;
						}
					}

					xtab[trueClass][prediction]++;
				}
			}
		}

#ifdef __linux__
		getrusage(RUSAGE_SELF, &usage);
		testTime +=
				((usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) - testTime);
#endif

		zOLoss.push_back(zeroOneLoss / static_cast<double>(count));
		rmse.push_back(sqrt(squaredError / count));
		rmsea.push_back(sqrt(squaredErrorAll / (count * noClasses)));
		logloss.push_back(-logLoss / count);
	}

	double variance = 0.0;

	for (unsigned int i = 0; i < noTestCases; i++) {
		double thisSumPHeqYsq = 0.0;
		for (CatValue y = 0; y < noClasses; y++) {
			double thisPHeqY = results[i][y]
					/ static_cast<double>(noExperiments);
			thisSumPHeqYsq += thisPHeqY * thisPHeqY;
		}
		variance += 0.5 * (1.0 - thisSumPHeqYsq);
	}

	variance /= noTestCases;

	printf("0-1 loss: ");
	print(zOLoss);
	printf("\nRMSE: ");
	print(rmse);
	printf("\nRMSE All Classes: ");
	print(rmsea);
	printf("\nLogarithmic Loss: ");
	print(logloss);
	if (noExperiments > 1) {
		double meanZOLoss = mean(zOLoss);
		printf(
				"\nMean 0-1 loss: %0.4f\nMean Bias: %.4f\nMean Variance: %0.4f\nMean RMSE: %0.4f\n\nMean RMSE All: %0.4f\nMean Logarithmic Loss: %0.4f\n",
				meanZOLoss, meanZOLoss - variance, variance, mean(rmse),
				mean(rmsea), mean(logloss));
	} else {
		putchar('\n');
	}

	for (unsigned int i = 0; i < noTestCases; i++) {
		delete[] results[i];
	}
	delete[] results;
}
