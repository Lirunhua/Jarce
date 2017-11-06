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
#include "MDLDiscretiser.h"
#include "utils.h"
#include <algorithm>
#include <stdio.h>

void MDLDiscretiser::FindCutPoints(std::vector<SubInstance> data,
		unsigned int begin, unsigned int end, double entropy, unsigned int k,
		std::vector<NumValue> &cuts) {
	unsigned int optimal_i;

	int N = end - begin;

	double entropy_s1 = 0, entropy_s2 = 0;
	double entropyLeftOptimal, entropyRightOptimal;

	unsigned int kLeft, kRight;
	unsigned int kLeftOptimal, kRightOptimal;

	NumValue candidate;
	NumValue candidateOptimal;

	double average_entropy = 0;
	double minimum_entropy = 0;

	std::vector<InstanceCount> classAll(noOfClasses_, 0);
	std::vector<InstanceCount> classLeft(noOfClasses_, 0);
	std::vector<InstanceCount> classRight(noOfClasses_, 0);

	if (N < 2)
		return;

	//compute classes info
	for (unsigned int i = begin; i < end; i++) {
		classAll[data[i].label]++;
	}

	//compute the candidates
	//determine which is the best

	int noCuts = 0;

	for (unsigned int i = begin; i < end - 1; i++) {

		classLeft[data[i].label]++;

		//if ((data[i].Attribute < data[i + 1].Attribute) && (data[i].label != data[i + 1].label)) {
                if ((data[i].Attribute < data[i + 1].Attribute)) {
			candidate = (data[i].Attribute + data[i + 1].Attribute) / 2;

			noCuts++;
			kLeft = 0;
			kRight = 0;

			//compute how many instances in each class for the two set
			for (unsigned int j = 0; j < noOfClasses_; j++) {
				classRight[j] = classAll[j] - classLeft[j];
				if (classLeft[j] > 0)
					kLeft++;
				if (classRight[j] > 0)
					kRight++;
			}

			//compute the entropy of the two sub set

			entropy_s1 = compEntropy(i + 1 - begin, classLeft);
			entropy_s2 = compEntropy(end - i - 1, classRight);

			//compute the average entropy
			average_entropy = ((i + 1 - begin) * entropy_s1
					+ (end - i - 1) * entropy_s2) / N;
			if (noCuts == 1) {
				minimum_entropy = average_entropy;
				candidateOptimal = candidate;
				kLeftOptimal = kLeft;
				kRightOptimal = kRight;
				entropyLeftOptimal = entropy_s1;
				entropyRightOptimal = entropy_s2;
				optimal_i = i + 1;
			} else if (average_entropy <= minimum_entropy) {
				minimum_entropy = average_entropy;
				candidateOptimal = candidate;
				kLeftOptimal = kLeft;
				kRightOptimal = kRight;
				entropyLeftOptimal = entropy_s1;
				entropyRightOptimal = entropy_s2;
				optimal_i = i + 1;
			}
//
//			if (floor(candidate * 1000000) == 954545) {
//
//				printf("weka:%f,%20.16f,%d,%d,%d\n", candidate, average_entropy,
//						i + 1, begin, end);
//				for (unsigned int k = i + 1 - 5; k < i + 1 + 5; k++)
//					printf("%d\t%f\t%d\n", k, data[k].Attribute, data[k].label);
//				printf("%d,%d\n",classLeft[0],classLeft[1]);
//				printf("%d,%d\n",classRight[0],classRight[1]);
//			}
//
//			if (candidate == 0.90454542636871338) {
//				printf("mdl:%f,%20.16f,%d,%d,%d\n", candidate, average_entropy,
//						i + 1, begin, end);
//				for (unsigned int k = i + 1 - 5; k < i + 1 + 5; k++)
//					printf("%d\t%f\t%d\n", k, data[k].Attribute, data[k].label);
//				printf("%d,%d\n",classLeft[0],classLeft[1]);
//				printf("%d,%d\n",classRight[0],classRight[1]);
//			}
		}

	}

	if (noCuts > 0) {
		double gain = 0, delta;

		//	delta = (log(pow(3, k) - 2) / log(2))
		delta = log2(pow(3.0, static_cast<int>(k)) - 2)
				- (k * entropy - kLeftOptimal * entropyLeftOptimal
						- kRightOptimal * entropyRightOptimal);
		gain = entropy - minimum_entropy;

		if (gain <= 0)
			return;
		//	if (gain > (((log(N - 1) / log(2)) / N) + delta / N)) {
		if (gain > ((log2(N - 1) / N) + delta / N)) {
			cuts.push_back(candidateOptimal);

			fflush(stdout);
			if (optimal_i - begin > 2)
				FindCutPoints(data, begin, optimal_i, entropyLeftOptimal,
						kLeftOptimal, cuts);
			if (end - optimal_i > 2)
				FindCutPoints(data, optimal_i, end, entropyRightOptimal,
						kRightOptimal, cuts);
		}
	}
}

double MDLDiscretiser::compEntropy(unsigned int noInst,
		const std::vector<InstanceCount> classCount) {
	double entropy;
	double pr_each_class;

//compute the entropy
	entropy = 0;
	for (unsigned int j = 0; j < noOfClasses_; j++) {
		if (classCount[j] > 0) {

			pr_each_class = (NumValue) classCount[j] / noInst;
			entropy += pr_each_class * log2(pr_each_class);
		}
	}

	entropy = -entropy;
	return entropy;

}

void MDLDiscretiser::discretise(std::vector<NumValue> &vals,
		const std::vector<CatValue> &classes, unsigned int noOfClasses, std::vector<NumValue> &cuts) {
        noOfClasses_ = noOfClasses;

	double entropy;
//	NumValue pr_each_class;
	std::vector<InstanceCount> classAll(noOfClasses_, 0);

	std::vector<SubInstance> data;
	std::vector<SubInstance>::const_iterator it;

	if (vals.size() != classes.size()) {
		printf("the sizes of vals and classes are not identical.\n");
		exit(0);
	}

	std::vector<NumValue>::const_iterator val_it;
	std::vector<CatValue>::const_iterator cla_it;

//combine the Attribute and the class vector as one vector
	for (val_it = vals.begin(), cla_it = classes.begin();
			val_it != vals.end() && cla_it != classes.end();
			val_it++, cla_it++) {
		SubInstance si(*val_it, *cla_it);
		data.push_back(si);
	}

//sort the data
	std::sort(data.begin(), data.end());

	// remove missing values from the array
	while (!data.empty() && data.back().Attribute == MISSINGNUM)
		data.pop_back();

//compute classes info
	for (it = data.begin(); it != data.end(); it++) {
		classAll[it->label]++;
	}

//	//output classes distribution
//		for (int i=0;i<classAll.size();i++) {
//			printf("%d\t%d\t%s\n",i,classAll[i],meta->classNames[i]);
//		}

//	FILEtype *f;
//	f = fopen("/home/shengleichen/Datasets/test/out.txt", "w");
//
//	for (int j = 0; j < data.size(); j++)
//		fprintf(f, "%f\t%d\n", data[j].Attribute, data[j].label);
//	fclose(f);

	entropy = compEntropy(data.size(), classAll);

	FindCutPoints(data, 0, data.size(), entropy, noOfClasses_, cuts);

//	//to sort the cut points
	std::sort(cuts.begin(), cuts.end());

	//printf("the cut points:\n");
//	std::vector<NumValue>::const_iterator cuts_it;
//	for (cuts_it = cuts.begin(); cuts_it != cuts.end(); cuts_it++) {
//		printf("%f\t", *cuts_it);
//	}
//	printf("\n");
//	fflush(stdout);
}
