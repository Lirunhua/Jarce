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

#include "aodeChen.h"
#include <assert.h>
#include "utils.h"
#include <algorithm>
#include "correlationMeasures.h"
#include "globals.h"
#include "utils.h"
#include "crosstab.h"


aodeChen::aodeChen(char* const *& argv, char* const * end) {
	name_ = "AODE";

	weighted = false;
	minCount = 100;
	subsumptionResolution = false;
	selected = false;
	su_ = false;
	mi_ = false;
	acmi_=false;
	chisq_ = false;
	empiricalMEst_ = false;
	empiricalMEst2_ = false;

	correlationFilter_=false;
	useThreshold_=false;
	threshold_=0;
	factor_=1.0;
	loo1_=false;
	loo2_=false;
	loo3_=false;
	loo4_=false;
	loo5_=false;
	loo6_=false;
	useAttribSelec_=false;

	attribSelected_=0;

	// get arguments
	while (argv != end) {
		if (*argv[0] != '-') {
			break;
		} else if (streq(argv[0] + 1, "loo1")) {
			loo1_ = true;
		} else if (streq(argv[0] + 1, "loo2")) {
			loo2_ = true;
		} else if (streq(argv[0] + 1, "loo3")) {
			loo3_ = true;
		} else if (streq(argv[0] + 1, "loo4")) {
			loo4_ = true;
		} else if (streq(argv[0] + 1, "loo5")) {
			loo5_ = true;
		} else if (streq(argv[0] + 1, "loo6")) {
			loo6_ = true;
		} else if (streq(argv[0] + 1, "empirical")) {
			empiricalMEst_ = true;
		} else if (streq(argv[0] + 1, "empirical2")) {
			empiricalMEst2_ = true;
		} else if (streq(argv[0] + 1, "sub")) {
			subsumptionResolution = true;
		} else if (argv[0][1] == 'n') {
			getUIntFromStr(argv[0] + 2, minCount, "n");
		} else if (streq(argv[0] + 1, "w")) {
			weighted = true;
		} else if (streq(argv[0] + 1, "selective")) {
			selected = true;
		} else if (streq(argv[0] + 1, "acmi")) {
			selected = true;
			acmi_ = true;
		} else if (streq(argv[0] + 1, "mi")) {
			selected = true;
			mi_ = true;
		} else if (argv[0][1] == 'a') {
			getUIntFromStr(argv[0] + 2, attribSelected_, "a");
			useAttribSelec_=true;
		} else if (argv[0][1] == 'f') {
			unsigned int factor;
			getUIntFromStr(argv[0] + 2, factor, "f");
			factor_ = factor / 10.0;
			while (factor_ >= 1)
				factor_ /= 10;
		}else if (streq(argv[0] + 1, "su")) {
			selected = true;
			su_ = true;
		}else if (streq(argv[0] + 1, "cf")) {
			correlationFilter_ = true;
		} else if (argv[0][1] == 't') {
			unsigned int thres;
			getUIntFromStr(argv[0] + 2, thres, "threshold");
			threshold_=thres/10.0;
			while(threshold_>=1)
				threshold_/=10;
			useThreshold_=true;
		} else if (streq(argv[0] + 1, "chisq")) {
			selected = true;
			chisq_ = true;
		} else {
			error("Aode does not support argument %s\n", argv[0]);
			break;
		}

		name_ += *argv;

		++argv;
	}
	if (selected == true) {
		if (mi_ == false && su_ == false && chisq_ == false)
			chisq_ = true;
	}
}

aodeChen::~aodeChen(void) {
}

void aodeChen::getCapabilities(capabilities &c) {
	c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void aodeChen::reset(InstanceStream &is) {
	xxyDist_.reset(is);
	inactiveCnt_ = 0;

	noCatAtts_ = is.getNoCatAtts();
	noClasses_ = is.getNoClasses();
	noChild_.resize(noCatAtts_,noCatAtts_);
	noAttSelected_=noCatAtts_;

	weight.assign(noCatAtts_, 1);

	instanceStream_ = &is;
	pass_ = 1;

	active_.assign(noCatAtts_, true);

	squaredError1D_.assign(noCatAtts_,0.0);

	//delete and initialise the rmse vector
	for (CategoricalAttribute x =0; x <squaredError_.size(); x++) {
		squaredError_[x].clear();
	}
	squaredError_.clear();
	squaredError_.resize(noCatAtts_);
	for (CategoricalAttribute x =0; x <squaredError_.size(); x++) {
		squaredError_[x].assign(noCatAtts_,0.0);
	}
}

void aodeChen::initialisePass() {

}

void aodeChen::train(const instance &inst) {

	if (pass_ == 1)
		xxyDist_.update(inst);
	else {
		assert(pass_ == 2);
		xxyDist_.remove(inst);
		LOOCV(inst);
		xxyDist_.add(inst);
	}
}

/// true iff no more passes are required. updated by finalisePass()
bool aodeChen::trainingIsFinished() {
	if (loo1_ == true || loo2_==true|| loo3_==true|| loo4_==true|| loo5_==true|| loo6_==true)
		return pass_ > 2;
	else
		return pass_ > 1;
}

// creates a comparator for two attributes based on their
//relative value with the class,such as mutual information, symmetrical uncertainty

class valCmpClass {
public:
	valCmpClass(std::vector<float> *s) {
		val = s;
	}

	bool operator()(CategoricalAttribute a, CategoricalAttribute b) {
		return (*val)[a] > (*val)[b];
	}

private:
	std::vector<float> *val;
};

int aodeChen::getNextElement(std::vector<CategoricalAttribute> &order,CategoricalAttribute ca, unsigned int noSelected) {
	CategoricalAttribute c = ca + 1;
	while (active_[order[c]] == false&&c < noSelected)
		c++;
	if (c < noSelected)
		return c;
	else
		return -1;
}

void aodeChen::finalisePass() {

	if(pass_==1)
	{
		if (loo1_== true ||loo2_== true|| loo3_==true|| loo5_==true|| loo6_==true) {

			orderedAtts.clear();
			for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
				orderedAtts.push_back(a);
			}
			//get the mutual information to rank the attrbiutes
			std::vector<float> measure;
			getMutualInformation(xxyDist_.xyCounts, measure);

			// sort the attributes on mutual information with the class

			if (!orderedAtts.empty()) {
				valCmpClass cmp(&measure);
				std::sort(orderedAtts.begin(), orderedAtts.end(), cmp);

				if (verbosity >= 2) {
					printf("The order of attributes ordered by the measure:\n");
					for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
						printf("%d:\t%f\t%u\n",  orderedAtts[a],measure[orderedAtts[a]],instanceStream_->getNoValues(orderedAtts[a]));
					}
				}
			}

		}
		if (loo4_==true ||loo6_==true) {


			orderedAttsForParent.resize(noCatAtts_);

			for (CategoricalAttribute parent = 0; parent< noCatAtts_; parent++) {
				for(CategoricalAttribute child = 0; child < noCatAtts_; child++)
					orderedAttsForParent[parent].push_back(child);
			}
			//get the mutual information to rank the attrbiutes
			crosstab<float> cmi(noCatAtts_);



			bool transpose=true;
			//getAttClassCondMutualInf(xxyDist_, cmi, transpose);

			// as this is not a good criterion,we try to use other conditional mutual information
			getCondMutualInf(xxyDist_, cmi);



			if (verbosity >= 2) {
				printf("The order of attributes ordered by the measure:\n");
			}
			// sort the attributes on mutual information with the class
			for (CategoricalAttribute parentIndex = 0; parentIndex < noCatAtts_;
					parentIndex++) {

				CategoricalAttribute parent;
				if(loo4_==true)
					parent=parentIndex;
				else
					parent= orderedAtts[parentIndex];

				valCmpClass cmp(&cmi[parent]);
				std::sort(orderedAttsForParent[parent].begin(),
						orderedAttsForParent[parent].end(), cmp);

				if (verbosity >= 2) {
					printf("%d\t",parent);
					const char * sep = "";
					for (CategoricalAttribute child = 0; child < noCatAtts_;
							child++)
					{
//						printf("%s%d:%f", sep,
//								orderedAttsForParent[parent][child],cmi[parent][orderedAttsForParent[parent][child]]);
						printf("%s%d", sep,
								orderedAttsForParent[parent][child]);
						if(child==0)
						sep = ", ";
					}
					printf("\n");
				}

			}
		}

		//	if (weighted) {
		//		weight.assign(noCatAtts_, 0);
		//		getMutualInformation(xxyDist_.xyCounts, weight);
		//
		//		//calculate the mutual information using count of instance rather than  m-esitmated probability
		//		if (verbosity >= 4) {
		//			for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
		//				printf("%f\n", weight[a]);
		//			}
		//		}
		//	}

		//calculate the mutual information using count of instance rather than  m-esitmated probability
		//
		//	if (weighted) {
		//
		//		weight.assign(noCatAtts_, 0);
		//
		//		for (CatValue i = 0; i < noCatAtts_; i++) {
		//
		//			for (CatValue x = 0; x <xxyDist_.instanceStream_->getNoValues(i); x++) {
		//
		//				for (CatValue y = 0; y <noClasses_; y++) {
		//
		//					InstanceCount countXy = xxyDist_.xyCounts.getCount(i,
		//							x,y);
		//					InstanceCount countY = xxyDist_.xyCounts.getClassCount(y);
		//					InstanceCount countX = xxyDist_.xyCounts.getCount(i,x);
		//					InstanceCount count = xxyDist_.xyCounts.count;
		//
		//
		//					if(countXy==0)
		//						continue;
		//					double weightXy=(double)countXy / count
		//							* log2(((double)countXy / countY) * ((double)count / countX));
		//					weight[i] +=weightXy ;
		//				}
		//			}
		//			//printf("%f\n",weight[i]);
		//		}
		//		//normalise(weight);
		//	}

		//calculate the mutual information using m-esitmated probability

		if (weighted) {

			weight.assign(noCatAtts_, 0);

			for (CatValue i = 0; i < noCatAtts_; i++) {

				for (CatValue x = 0; x < xxyDist_.getNoValues(i); x++) {
					for (CatValue y = 0; y < noClasses_; y++) {
						double pXy = xxyDist_.xyCounts.jointP(i, x, y);
						double pY = xxyDist_.xyCounts.p(y);

						double pX = 0;
						for (CatValue yPrime = 0; yPrime < noClasses_; yPrime++) {
							pX += xxyDist_.xyCounts.jointP(i, x, yPrime);
						}

						if (pXy == 0)
							continue;
						double weightXy = pXy * log2(pXy / (pX * pY));
						weight[i] += weightXy;
					}
				}
			}
		}

		if (verbosity >= 4) {
			for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
				printf("%f\n", weight[a]);
			}
		}

		if (selected) {

			std::vector<CategoricalAttribute> order;

			for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
				order.push_back(a);
			}

			if (mi_ == true || su_ == true || acmi_==true) {
				//calculate the symmetrical uncertainty between each attribute and class
				std::vector<float> measure;
				crosstab<float> acmi(noCatAtts_);

				if (mi_ == true)
					getMutualInformation(xxyDist_.xyCounts, measure);
				else if (su_ == true)
					getSymmetricalUncert(xxyDist_.xyCounts, measure);
				else if (acmi_ == true) {
					getAttClassCondMutualInf(xxyDist_, acmi);
					measure.assign(noCatAtts_, 0.0);

					for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
						for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

							// the way to sum acmi has been corrected.
							measure[x1] += acmi[x1][x2];
							measure[x2] += acmi[x2][x1];

							if(verbosity>=2)
							{
								if(x1==2)
								printf("%u,",x2);
								if(x2==2)
									printf("%u,",x1);
							}
						}


					}
				}

				if (verbosity >= 2) {
					if (mi_ == true)
						printf("Selecting according to mutual information:\n");
					else if (su_ == true)
						printf("Selecting according to symmetrical uncertainty:\n");
					else if (acmi_ == true)
						printf("Selecting according to attribute and class conditional mutual information:\n");
					print(measure);
					printf("\n");
				}

				// sort the attributes on symmetrical uncertainty with the class

				if (!order.empty()) {
					valCmpClass cmp(&measure);

					std::sort(order.begin(), order.end(), cmp);

					if (verbosity >= 2) {
						const char * sep = "";
						printf("The order of attributes ordered by the measure:\n");
						for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
							printf("%d:\t%f\t%u\n",  order[a],measure[order[a]],instanceStream_->getNoValues(order[a]));
							sep = ", ";
						}
						printf("\n");

					}
				}

				//select half of the attributes as spodes default

				unsigned int noSelected; ///< the number of selected attributes


				// set the attribute selected or unselected

				if (useThreshold_ == true) {
					noSelected = noCatAtts_;
					for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
						if (measure[a] < threshold_) {
							active_[a] = false;
							noSelected--;
						}

					}
					if (verbosity >= 2) {
						const char * sep = "";
						printf(
								"The attributes being selected according to the threshold:\n");
						for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
							if (measure[a] >= threshold_) {
								printf("%s%d", sep, order[a]);
								sep = ", ";
							}
						}
						printf("\n");
					}
				} else if (useAttribSelec_ == true) {
					noSelected = 1;
					if (attribSelected_ >= noCatAtts_)
						error("the attribute is out of range!\n");

					for (CategoricalAttribute a = 0; a <noCatAtts_; a++) {
						if (a != attribSelected_) {
							active_[a] = false;
						}
					}
					printf(
							"The user specified attribute is :  %u\n",
							attribSelected_);

				}
				else {

					noSelected = static_cast<unsigned int>(noCatAtts_ * factor_);

					for (CategoricalAttribute a = noSelected; a < noCatAtts_; a++) {
						active_[order[a]] = false;
					}

					if (verbosity >= 2) {
						const char * sep = "";
						printf(
								"The attributes being selected according to the number:\n");
						for (CategoricalAttribute a = 0; a < noSelected; a++) {
							printf("%s%d", sep, order[a]);
							sep = ", ";
						}
						printf("\n");
					}

				}

				if (correlationFilter_ == true) {
					if (!(su_ == true && mi_ == false)) {
						printf(
								"Correlation filter can only be used with symmetrical uncertainty!\n");
						return;
					}
					int Fp = 0;
					int Fq;

					do {
						Fq = getNextElement(order, Fp, noSelected);

						if (Fq != -1) {
							do {

								double SUpq = getSymmetricalUncert(xxyDist_,
										order[Fp], order[Fq]);

								if (SUpq >= measure[order[Fq]]) {
									active_[order[Fq]] = false;
								}
								Fq = getNextElement(order, Fq, noSelected);

							} while (Fq != -1);

						}
						Fp = getNextElement(order, Fp, noSelected);
					} while (Fp != -1);

					printf(
							"The following attributes have been selected by correlation filter:\n");
					for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
						if (active_[order[a]])
							printf("%d,%f\t", order[a], measure[order[a]]);
					}
					printf("\n");
				}

			} else if (chisq_ == true) {

				bool flag = true;
				double lowest;
				CategoricalAttribute attLowest;

				for (std::vector<CategoricalAttribute>::const_iterator it =
						order.begin(); it != order.end(); it++) {

					CategoricalAttribute a = *it;
					const unsigned int rows = instanceStream_->getNoValues(a);

					if (rows < 2) {
						active_[a] = false;
						inactiveCnt_++;
					} else {
						const unsigned int cols = noClasses_;
						InstanceCount *tab;
						allocAndClear(tab, rows * cols);

						for (CatValue r = 0; r < rows; r++) {

							for (CatValue c = 0; c < cols; c++) {
								tab[r * cols + c] += xxyDist_.xyCounts.getCount(a,
										r, c);

								if (verbosity >= 2) {

									if (a == 2) {
										printf("%d ", tab[r * cols + c]);
									}

								}

							}
							if (verbosity >= 2) {
								if (a == 2) {
									printf("\n");
								}
							}
						}

						double critVal = 0.05 / noCatAtts_;
						//double critVal = 0.0000000005 / noCatAtts_;
						double chisqVal = chiSquare(tab, rows, cols);

						if (verbosity >= 2){

							printf("the chi-square value of attribute %s: %40.40f\n",instanceStream_->getCatAttName(a), chisqVal);

						}

						//select the attribute with lowest chisq value as parent if there is attribute satisfying the
						//significance level of 5%
						if (flag == true) {
							lowest = chisqVal;
							attLowest = a;

							flag = false;
						} else {
							if (lowest > chisqVal) {
								lowest = chisqVal;
								attLowest = a;
							}
						}

						if (chisqVal > critVal) {

							if (verbosity >= 2)
								printf(
										"%s suppressed by chisq test against class\n",
										instanceStream_->getCatAttName(a));
							active_[a] = false;
							inactiveCnt_++;
						}
						delete[] tab;
					}
				}
				if (inactiveCnt_ == noCatAtts_) {
					active_[attLowest] = true;
					if (verbosity >= 2)
						printf("Only the attribute %u is active.\n", attLowest);

				}
				if (verbosity >= 2)
					printf(
							"The number of active parent and total attributes are: %u,%u\n",
							noCatAtts_ - inactiveCnt_, noCatAtts_);
			}
		}

	}
	else if(pass_==2)
	{
		if (loo1_ == true) {
			printf("the first loocv selection.\n");
		} else if (loo2_ == true) {
			printf("the second loocv selection.\n");
		} else if (loo3_ == true) {
			printf("the third loocv selection.\n");
		} else if (loo4_ == true) {
			printf("the fourth loocv selection.\n");
		} else if (loo5_ == true) {
			printf("the fifth loocv selection.\n");
		}else if (loo6_ == true) {
			printf("the sixth loocv selection.\n");
		}


		if(loo1_==true||loo2_==true||loo3_==true)
		{
			for (CategoricalAttribute att = 0; att < noCatAtts_;
					att++) {
				squaredError1D_[att] = sqrt(
						squaredError1D_[att] / xxyDist_.xyCounts.count);
			}
//			noAttSelected_=indexOfMinVal(squaredError1D_)+1;
			print(squaredError1D_);
			printf("\n");
			printf("The best model is: %u in %u\n",noAttSelected_,noCatAtts_);
		}else if(loo5_==true||loo6_==true)
		{
			for (CategoricalAttribute parent = 0; parent < noCatAtts_;
					parent++) {
				for (CategoricalAttribute child = 0; child < noCatAtts_;
						child++) {
					squaredError_[parent][child] = sqrt(
							squaredError_[parent][child]
									/ xxyDist_.xyCounts.count);
				}
			}

			optParentIndex_=0;
//			optChildIndex_=indexOfMinVal(squaredError_[optParentIndex_]);
			double minVal=squaredError_[optParentIndex_][optChildIndex_];

			for (CatValue parent = 1; parent < noCatAtts_; parent++) {
//				unsigned int optRow=indexOfMinVal(squaredError_[parent]);
//				double  minValRow=squaredError_[parent][optRow];
//				if(minVal>minValRow)
//				{
//					optParentIndex_=parent;
//					optChildIndex_=optRow;
//					minVal=minValRow;
//				}
			}
			optParentIndex_++;
			optChildIndex_++;
			printf("The best model is:(%u,%u) in (%u,%u).\n",optParentIndex_,optChildIndex_,noCatAtts_,noCatAtts_);

			if(verbosity>=2)
			{
				for (CatValue parent = 0; parent < noCatAtts_; parent++) {
					print(squaredError_[parent]);
					printf("\n");
				}
			}
		}
		else
		{
			assert(loo4_==true);
			printf("The best model for each parent is:\n");
			for (CatValue parent = 0; parent < noCatAtts_; parent++) {
//				noChild_[parent] = indexOfMinVal(squaredError_[parent]) + 1;
//				printf("parent \t %u, %u, ",parent,noChild_[parent]);
//				print(squaredError_[parent]);
//				printf("\n");
			}
			printf("\n");
		}
	}
	++pass_;
}

void aodeChen::LOOCV(const instance &inst)
{
	const InstanceCount totalCount = xxyDist_.xyCounts.count;
	std::vector< std::vector < double > > classDist;

	classDist.resize(noCatAtts_);
	for (CatValue y = 0; y < noCatAtts_; y++) {
		classDist[y].assign(noClasses_,0.0);
	}
	const CatValue trueClass = inst.getClass();

	// scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max() / noCatAtts_;
	 scaleFactor =1;

	CatValue delta = 0;

	//try to increase the efficiency

	fdarray<double> spodeProbs(noCatAtts_, noClasses_);
	fdarray<InstanceCount> xyCount(noCatAtts_, noClasses_);
	std::vector<bool> active(noCatAtts_, false);


	if(loo1_==true)
	{

		for (CatValue parent = 0; parent < noCatAtts_; parent++) {

				const CatValue parentVal = inst.getCatVal(parent);

				for (CatValue y = 0; y < noClasses_; y++) {
					xyCount[parent][y] = xxyDist_.xyCounts.getCount(parent,
							parentVal, y);
				}

				if (active_[parent]) {
					if (xxyDist_.xyCounts.getCount(parent, parentVal) > 0) {
						delta++;
						active[parent] = true;

						if (empiricalMEst_) {
							for (CatValue y = 0; y < noClasses_; y++) {
								spodeProbs[parent][y] = weight[parent]
										* empiricalMEstimate(xyCount[parent][y],
												totalCount,
												xxyDist_.xyCounts.p(y)
														* xxyDist_.xyCounts.p(
																parent, parentVal))
										* scaleFactor;
							}
						} else {
							for (CatValue y = 0; y < noClasses_; y++) {
								spodeProbs[parent][y] = weight[parent]
										* mEstimate(xyCount[parent][y], totalCount,
												noClasses_
														* xxyDist_.getNoValues(
																parent))
										* scaleFactor;
								if(verbosity>=3)
								{

									printf("%u,%u,%f\n",parent,y,spodeProbs[parent][y]);
								}

							}
						}
					} else if (verbosity >= 5)
						printf("%d\n", parent);
				}
			}

		if (delta == 0) {
			//nbClassify(inst, classDist, xxyDist_.xyCounts);
			printf("there are no eligible parents.\n");
			exit(0);
		}

		if(verbosity>=2)
		{
			for (CatValue parent = 0; parent < noCatAtts_; parent++) {
				for (CatValue y = 0; y < noClasses_; y++) {
					printf("%f,",spodeProbs[parent][y]);
				}
				printf("  %u\n",parent);

			}
		}
		std::vector<std::vector<std::vector<double> > > model;
		model.resize(noCatAtts_);

		for (CategoricalAttribute parent = 0; parent < noCatAtts_;
				parent++) {

			model[parent].resize(noCatAtts_);
			if (active[parent] == true) {

				const CatValue parentVal = inst.getCatVal(parent);
				for (CategoricalAttribute childIndex = 0; childIndex < noCatAtts_; childIndex++) {

					CategoricalAttribute child = orderedAtts[childIndex];
					model[parent][childIndex].resize(noClasses_);
					for (CatValue y = 0; y < noClasses_; y++) {

						CatValue childVal = inst.getCatVal(child);
						if (child != parent) {
							spodeProbs[parent][y] *= xxyDist_.p(child, childVal,
									parent, parentVal, y);
						}
						model[parent][childIndex][y] = spodeProbs[parent][y];
					}
	//				normalise(model[parent][child]);
	//				const double error = 1.0 - model[parent][child][trueClass];
	//				squaredError_[parent][child] += error * error;

				}
			}
		}

		if (verbosity >= 3) {
			printf("true class is %u\n", trueClass);
		}
		for (CatValue parent = 0; parent < noCatAtts_; parent++) {
			if (active[parent]) {
				for (CategoricalAttribute childIndex = 0;
						childIndex < noCatAtts_; childIndex++) {
					for (CatValue y = 0; y < noClasses_; y++) {
						classDist[childIndex][y] +=
								model[parent][childIndex][y];
					}
				}
			}
		}

		for (CategoricalAttribute childIndex = 0; childIndex < noCatAtts_; childIndex++) {

			CategoricalAttribute child = orderedAtts[childIndex];
			normalise(classDist[childIndex]);
			if (verbosity >= 2) {
				print(classDist[childIndex]);
				printf("\t%u,%u\n", child,indexOfMaxVal(classDist[childIndex]));
			}
			//		const CatValue prediction = indexOfMaxVal(classDist[child]);
			//		if (prediction != trueClass)
			//			zeroOneLoss_[child]++;
			const double error = 1.0 - classDist[childIndex][trueClass];
			squaredError1D_[childIndex] += error * error;
		}
		if (verbosity >= 2) {
			print(squaredError1D_);
			printf("\n");
		}
	}

	else if(loo2_==true)
	{
		// initial spode assignment of joint probability of parent and class
		for (CatValue parentIndex = 0; parentIndex < noCatAtts_; parentIndex++) {
			CategoricalAttribute parent = orderedAtts[parentIndex];
			const CatValue parentVal = inst.getCatVal(parent);
			for (CatValue y = 0; y < noClasses_; y++) {
				xyCount[parent][y] = xxyDist_.xyCounts.getCount(parent,
						parentVal, y);
			}

			if (active_[parent]) {
				if (xxyDist_.xyCounts.getCount(parent, parentVal) > 0) {
					delta++;
					active[parent] = true;

					if (empiricalMEst_) {
						for (CatValue y = 0; y < noClasses_; y++) {
							spodeProbs[parentIndex][y] = weight[parent]
									* empiricalMEstimate(xyCount[parent][y],
											totalCount,	xxyDist_.xyCounts.p(y)
													* xxyDist_.xyCounts.p(parent, parentVal))
									* scaleFactor;
						}
					} else {
						for (CatValue y = 0; y < noClasses_; y++) {
							spodeProbs[parentIndex][y] = weight[parent]
									* mEstimate(xyCount[parent][y], totalCount,
											noClasses_* xxyDist_.getNoValues(parent))
									* scaleFactor;
							if (verbosity >= 3) {
								printf("%u,%u,%f\n", parent, y,
										spodeProbs[parent][y]);
							}
						}
					}
				} else if (verbosity >= 5)
					printf("%d\n", parent);
			}
		}

		if (delta == 0) {
			printf("there are no eligible parents.\n");
			exit(0);
		}

		for (CategoricalAttribute parentIndex = 0; parentIndex < noCatAtts_;
				parentIndex++) {
			CategoricalAttribute parent = orderedAtts[parentIndex];
			if (active[parent] == true) {

				const CatValue parentVal = inst.getCatVal(parent);
				for (CategoricalAttribute childIndex = 0; childIndex < noCatAtts_; childIndex++) {

					CategoricalAttribute child = orderedAtts[childIndex];
					for (CatValue y = 0; y < noClasses_; y++) {

						CatValue childVal = inst.getCatVal(child);
						if (child != parent) {
							spodeProbs[parentIndex][y] *= xxyDist_.p(child, childVal,
									parent, parentVal, y);
						}
						//model[parentIndex][childIndex][y] = spodeProbs[parentIndex][y];
					}
				}
			}
		}
		if (verbosity >= 3) {
			printf("true class is %u\n", trueClass);
		}

		std::vector<double> spodeProbsSumOnRow;
		spodeProbsSumOnRow.resize(noClasses_,0.0);

		for (CategoricalAttribute parentIndex = 0; parentIndex < noCatAtts_;
				parentIndex++) {
			for (CatValue y = 0; y < noClasses_; y++) {
				spodeProbsSumOnRow[y]+= spodeProbs[parentIndex][y];
				classDist[parentIndex][y]=spodeProbsSumOnRow[y];
			}
			normalise(classDist[parentIndex]);
			const double error = 1.0 - classDist[parentIndex][trueClass];
			squaredError1D_[parentIndex] += error * error;
		}

		if(verbosity>=3)
		{
			for (CatValue parent = 0; parent < noCatAtts_; parent++) {
				for (CatValue y = 0; y < noClasses_; y++) {
					printf("%f,",spodeProbs[parent][y]);
				}
				printf("  %u\n",parent);

			}
		}
		if(verbosity>=3)
		{
			for (CatValue parent = 0; parent < noCatAtts_; parent++) {
				for (CatValue y = 0; y < noClasses_; y++) {
					printf("%f,",classDist[parent][y]);
				}
				printf("  %u\n",parent);

			}
		}
	}
	else if(loo3_==true)
	{

		// initial spode assignment of joint probability of parent and class
		for (CatValue parentIndex = 0; parentIndex < noCatAtts_; parentIndex++) {
			CategoricalAttribute parent = orderedAtts[parentIndex];
			const CatValue parentVal = inst.getCatVal(parent);
			for (CatValue y = 0; y < noClasses_; y++) {
				xyCount[parent][y] = xxyDist_.xyCounts.getCount(parent,
						parentVal, y);
			}

			if (active_[parent]) {
				if (xxyDist_.xyCounts.getCount(parent, parentVal) > 0) {
					delta++;
					active[parent] = true;

					if (empiricalMEst_) {
						for (CatValue y = 0; y < noClasses_; y++) {
							spodeProbs[parentIndex][y] = weight[parent]
									* empiricalMEstimate(xyCount[parent][y],
											totalCount,	xxyDist_.xyCounts.p(y)
													* xxyDist_.xyCounts.p(parent, parentVal))
									* scaleFactor;
						}
					} else {
						for (CatValue y = 0; y < noClasses_; y++) {
							spodeProbs[parentIndex][y] = weight[parent]
									* mEstimate(xyCount[parent][y], totalCount,
											noClasses_* xxyDist_.getNoValues(parent))
									* scaleFactor;
							if (verbosity >= 3) {
								printf("%u,%u,%f\n", parent, y,
										spodeProbs[parent][y]);
							}
						}
					}
				} else if (verbosity >= 5)
					printf("%d\n", parent);
			}
		}

		if (delta == 0) {
			printf("there are no eligible parents.\n");
			exit(0);
		}

		std::vector<std::vector<std::vector<double> > > model;
		model.resize(noCatAtts_);

		for (CategoricalAttribute parentIndex = 0; parentIndex < noCatAtts_;
				parentIndex++) {

			CategoricalAttribute parent = orderedAtts[parentIndex];
			model[parentIndex].resize(noCatAtts_);
			if (active[parent] == true) {

				const CatValue parentVal = inst.getCatVal(parent);
				for (CategoricalAttribute childIndex = 0; childIndex < noCatAtts_; childIndex++) {

					CategoricalAttribute child = orderedAtts[childIndex];
					model[parentIndex][childIndex].resize(noClasses_);
					for (CatValue y = 0; y < noClasses_; y++) {

						CatValue childVal = inst.getCatVal(child);
						if (child != parent) {
							spodeProbs[parentIndex][y] *= xxyDist_.p(child, childVal,
									parent, parentVal, y);
						}
						model[parentIndex][childIndex][y] = spodeProbs[parentIndex][y];
					}
				}
			}
		}
		if (verbosity >= 3) {
			printf("true class is %u\n", trueClass);
		}

		for (CategoricalAttribute child = 0; child < noCatAtts_; child++) {
			for (CatValue parent = 0; parent <= child; parent++) {
				if (active[orderedAtts[parent]]) {
					for (CatValue y = 0; y < noClasses_; y++) {
						classDist[child][y] +=
								model[parent][child][y];
					}
				}
			}
			normalise(classDist[child]);
			const double error = 1.0 - classDist[child][trueClass];
			squaredError1D_[child] += error * error;
		}
	}
	else if(loo4_==true)
	{
		for (CatValue parent = 0; parent < noCatAtts_; parent++) {

				const CatValue parentVal = inst.getCatVal(parent);

				for (CatValue y = 0; y < noClasses_; y++) {
					xyCount[parent][y] = xxyDist_.xyCounts.getCount(parent,
							parentVal, y);
				}

				if (active_[parent]) {
					if (xxyDist_.xyCounts.getCount(parent, parentVal) > 0) {
						delta++;
						active[parent] = true;

						if (empiricalMEst_) {
							for (CatValue y = 0; y < noClasses_; y++) {
								spodeProbs[parent][y] = weight[parent]
										* empiricalMEstimate(xyCount[parent][y],
												totalCount,
												xxyDist_.xyCounts.p(y)
														* xxyDist_.xyCounts.p(
																parent, parentVal))
										* scaleFactor;
							}
						} else {
							for (CatValue y = 0; y < noClasses_; y++) {
								spodeProbs[parent][y] = weight[parent]
										* mEstimate(xyCount[parent][y], totalCount,
												noClasses_
														* xxyDist_.getNoValues(
																parent))
										* scaleFactor;
								if(verbosity>=3)
								{

									printf("%u,%u,%f\n",parent,y,spodeProbs[parent][y]);
								}

							}
						}
					} else if (verbosity >= 5)
						printf("%d\n", parent);
				}
			}

		if (delta == 0) {
			//nbClassify(inst, classDist, xxyDist_.xyCounts);
			printf("there are no eligible parents.\n");
			exit(0);
		}

		if(verbosity>=3)
		{
			for (CatValue parent = 0; parent < noCatAtts_; parent++) {
				for (CatValue y = 0; y < noClasses_; y++) {
					printf("%f,",spodeProbs[parent][y]);
				}
				printf("  %u\n",parent);

			}
		}
		std::vector<std::vector<std::vector<double> > > model;
		model.resize(noCatAtts_);

		for (CategoricalAttribute parent = 0; parent < noCatAtts_;
				parent++) {

			model[parent].resize(noCatAtts_);
			if (active[parent] == true) {

				const CatValue parentVal = inst.getCatVal(parent);
				for (CategoricalAttribute childIndex = 0; childIndex < noCatAtts_; childIndex++) {

					CategoricalAttribute child = orderedAttsForParent[parent][childIndex];
					if(verbosity>=4)
						printf("%d\n",child);
					model[parent][childIndex].resize(noClasses_);
					for (CatValue y = 0; y < noClasses_; y++) {

						CatValue childVal = inst.getCatVal(child);
						if (child != parent) {
							spodeProbs[parent][y] *= xxyDist_.p(child, childVal,
									parent, parentVal, y);
						}
						model[parent][childIndex][y] = spodeProbs[parent][y];
					}
	//				normalise(model[parent][child]);
	//				const double error = 1.0 - model[parent][child][trueClass];
	//				squaredError_[parent][child] += error * error;

				}
			}
		}
		for (CategoricalAttribute parent = 0; parent < noCatAtts_; parent++) {
			if (active[parent] == true) {
				for (CategoricalAttribute childIndex = 0;
						childIndex < noCatAtts_; childIndex++) {
					//CategoricalAttribute child = orderedAttsForParent[parent][childIndex];

					normalise(model[parent][childIndex]);
					const double error = 1.0 - model[parent][childIndex][trueClass];
					squaredError_[parent][childIndex] += error * error;
				}
			}
		}
	}
	else if(loo5_==true|| loo6_==true)
	{

		// initial spode assignment of joint probability of parent and class
		for (CatValue parentIndex = 0; parentIndex < noCatAtts_; parentIndex++) {
			CategoricalAttribute parent = orderedAtts[parentIndex];
			const CatValue parentVal = inst.getCatVal(parent);
			for (CatValue y = 0; y < noClasses_; y++) {
				xyCount[parent][y] = xxyDist_.xyCounts.getCount(parent,
						parentVal, y);
			}

			if (active_[parent]) {
				if (xxyDist_.xyCounts.getCount(parent, parentVal) > 0) {
					delta++;
					active[parent] = true;

					if (empiricalMEst_) {
						for (CatValue y = 0; y < noClasses_; y++) {
							spodeProbs[parentIndex][y] = weight[parent]
									* empiricalMEstimate(xyCount[parent][y],
											totalCount,	xxyDist_.xyCounts.p(y)
													* xxyDist_.xyCounts.p(parent, parentVal))
									* scaleFactor;
						}
					} else {
						for (CatValue y = 0; y < noClasses_; y++) {
							spodeProbs[parentIndex][y] = weight[parent]
									* mEstimate(xyCount[parent][y], totalCount,
											noClasses_* xxyDist_.getNoValues(parent))
									* scaleFactor;
							if (verbosity >= 3) {
								printf("%u,%u,%f\n", parent, y,
										spodeProbs[parent][y]);
							}
						}
					}
				} else if (verbosity >= 5)
					printf("%d\n", parent);
			}
		}

		if (delta == 0) {
			printf("there are no eligible parents.\n");
			exit(0);
		}

		std::vector<std::vector<std::vector<double> > > model;
		model.resize(noCatAtts_);


		for (CategoricalAttribute parentIndex = 0; parentIndex < noCatAtts_;
				parentIndex++) {

			CategoricalAttribute parent = orderedAtts[parentIndex];
			model[parentIndex].resize(noCatAtts_);
			if (active[parent] == true) {

				const CatValue parentVal = inst.getCatVal(parent);
				for (CategoricalAttribute childIndex = 0; childIndex < noCatAtts_; childIndex++) {

					CategoricalAttribute child;
					if(loo5_==true)
						child= orderedAtts[childIndex];
					else
					{
						assert(loo6_==true);
						child = orderedAttsForParent[parent][childIndex];
					}


					model[parentIndex][childIndex].resize(noClasses_);
					for (CatValue y = 0; y < noClasses_; y++) {

						CatValue childVal = inst.getCatVal(child);
						if (child != parent) {
							spodeProbs[parentIndex][y] *= xxyDist_.p(child, childVal,
									parent, parentVal, y);
						}
						model[parentIndex][childIndex][y] = spodeProbs[parentIndex][y];
					}
				}
			}
		}
		if (verbosity >= 3) {
			printf("true class is %u\n", trueClass);
		}

		for (CategoricalAttribute childIndex = 0; childIndex < noCatAtts_;
				childIndex++) {

			std::vector<double> spodeProbsSumOnRow;
			spodeProbsSumOnRow.resize(noClasses_, 0.0);

			for (CategoricalAttribute parentIndex = 0; parentIndex < noCatAtts_;
					parentIndex++) {

				for (CatValue y = 0; y < noClasses_; y++) {
					spodeProbsSumOnRow[y] += model[parentIndex][childIndex][y];
					classDist[parentIndex][y] = spodeProbsSumOnRow[y];
				}
				normalise(classDist[parentIndex]);
				const double error = 1.0 - classDist[parentIndex][trueClass];
				squaredError_[parentIndex][childIndex] += error * error;
			}
		}

		if(verbosity>=3)
		{
			for (CatValue parent = 0; parent < noCatAtts_; parent++) {
				print(squaredError_[parent]);
				printf("\n");
			}
		}
	}

}


void aodeChen::classify(const instance &inst, std::vector<double> &classDist) {
	std::vector<bool> generalizationSet;
	const InstanceCount totalCount = xxyDist_.xyCounts.count;

	for (CatValue y = 0; y < noClasses_; y++)
		classDist[y] = 0;

	// scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max() / noCatAtts_;

	CatValue delta = 0;

	//try to increase the efficiency

	fdarray<double> spodeProbs(noCatAtts_, noClasses_);
	fdarray<InstanceCount> xyCount(noCatAtts_, noClasses_);
	std::vector<bool> active(noCatAtts_, false);


	if(loo2_==true)
	{
		CategoricalAttribute noParent=noAttSelected_;
		for (CatValue parentIndex = 0; parentIndex < noParent; parentIndex++) {
			CategoricalAttribute parent = orderedAtts[parentIndex];
			const CatValue parentVal = inst.getCatVal(parent);
			for (CatValue y = 0; y < noClasses_; y++) {
				xyCount[parent][y] = xxyDist_.xyCounts.getCount(parent,
						parentVal, y);
			}

			if (active_[parent]) {
				if (xxyDist_.xyCounts.getCount(parent, parentVal) > 0) {
					delta++;
					active[parent] = true;

					if (empiricalMEst_) {
						for (CatValue y = 0; y < noClasses_; y++) {
							spodeProbs[parentIndex][y] = weight[parent]
									* empiricalMEstimate(xyCount[parent][y],
											totalCount,
											xxyDist_.xyCounts.p(y)
													* xxyDist_.xyCounts.p(
															parent, parentVal))
									* scaleFactor;
						}
					} else {
						for (CatValue y = 0; y < noClasses_; y++) {
							spodeProbs[parentIndex][y] = weight[parent]
									* mEstimate(xyCount[parent][y], totalCount,
											noClasses_
													* xxyDist_.getNoValues(
															parent))
									* scaleFactor;
							if (verbosity >= 3) {

								printf("%u,%u,%f\n", parent, y,
										spodeProbs[parent][y]);
							}

						}
					}
				} else if (verbosity >= 5)
					printf("%d\n", parent);
			}
		}

		if (delta == 0) {
			nbClassify(inst, classDist, xxyDist_.xyCounts);
			return;
		}

		for (CatValue parentIndex = 0; parentIndex < noParent; parentIndex++) {
			CategoricalAttribute parent = orderedAtts[parentIndex];
			const CatValue parentVal = inst.getCatVal(parent);
			if (active[parent] == true) {

				for (CategoricalAttribute childIndex = 0;
						childIndex < noCatAtts_; childIndex++) {

					CategoricalAttribute child = orderedAtts[childIndex];
					for (CatValue y = 0; y < noClasses_; y++) {
						if (child != parent) {
							CatValue childVal = inst.getCatVal(child);
							spodeProbs[parentIndex][y] *= xxyDist_.p(child,
									childVal, parent, parentVal, y);
						}
					}
				}
			}
		}

		for (CatValue parent = 0; parent < noParent; parent++) {
			if (active[orderedAtts[parent]]) {
				for (CatValue y = 0; y < noClasses_; y++) {
					classDist[y] += spodeProbs[parent][y];
				}
			}
		}
		normalise(classDist);
		return;
	}
	// end of classify in loocv2


	if(loo3_==true)
	{
		//CategoricalAttribute noParent=noAttSelected_;
		for (CatValue parentIndex = 0; parentIndex < noAttSelected_; parentIndex++) {
			CategoricalAttribute parent = orderedAtts[parentIndex];
			const CatValue parentVal = inst.getCatVal(parent);
			for (CatValue y = 0; y < noClasses_; y++) {
				xyCount[parent][y] = xxyDist_.xyCounts.getCount(parent,
						parentVal, y);
			}

			if (active_[parent]) {
				if (xxyDist_.xyCounts.getCount(parent, parentVal) > 0) {
					delta++;
					active[parent] = true;

					if (empiricalMEst_) {
						for (CatValue y = 0; y < noClasses_; y++) {
							spodeProbs[parentIndex][y] = weight[parent]
									* empiricalMEstimate(xyCount[parent][y],
											totalCount,
											xxyDist_.xyCounts.p(y)
													* xxyDist_.xyCounts.p(
															parent, parentVal))
									* scaleFactor;
						}
					} else {
						for (CatValue y = 0; y < noClasses_; y++) {
							spodeProbs[parentIndex][y] = weight[parent]
									* mEstimate(xyCount[parent][y], totalCount,
											noClasses_
													* xxyDist_.getNoValues(
															parent))
									* scaleFactor;
							if (verbosity >= 3) {

								printf("%u,%u,%f\n", parent, y,
										spodeProbs[parent][y]);
							}

						}
					}
				} else if (verbosity >= 5)
					printf("%d\n", parent);
			}
		}

		if (delta == 0) {
			nbClassify(inst, classDist, xxyDist_.xyCounts);
			return;
		}

		for (CatValue parentIndex = 0; parentIndex < noAttSelected_; parentIndex++) {
			CategoricalAttribute parent = orderedAtts[parentIndex];
			const CatValue parentVal = inst.getCatVal(parent);
			if (active[parent] == true) {

				for (CategoricalAttribute childIndex = 0;
						childIndex < noAttSelected_; childIndex++) {

					CategoricalAttribute child = orderedAtts[childIndex];
					for (CatValue y = 0; y < noClasses_; y++) {
						if (child != parent) {
							CatValue childVal = inst.getCatVal(child);
							spodeProbs[parentIndex][y] *= xxyDist_.p(child,
									childVal, parent, parentVal, y);
						}
					}
				}
			}
		}

		for (CatValue parent = 0; parent < noAttSelected_; parent++) {
			if (active[orderedAtts[parent]]) {
				for (CatValue y = 0; y < noClasses_; y++) {
					classDist[y] += spodeProbs[parent][y];
				}
			}
		}
		normalise(classDist);
		return;
	}
	// end of classify in loocv3


	if(loo5_==true||loo6_==true)
	{
		//CategoricalAttribute noParent=noAttSelected_;
		for (CatValue parentIndex = 0; parentIndex < noAttSelected_; parentIndex++) {
			CategoricalAttribute parent = orderedAtts[parentIndex];
			const CatValue parentVal = inst.getCatVal(parent);
			for (CatValue y = 0; y < noClasses_; y++) {
				xyCount[parent][y] = xxyDist_.xyCounts.getCount(parent,
						parentVal, y);
			}

			if (active_[parent]) {
				if (xxyDist_.xyCounts.getCount(parent, parentVal) > 0) {
					delta++;
					active[parent] = true;

					if (empiricalMEst_) {
						for (CatValue y = 0; y < noClasses_; y++) {
							spodeProbs[parentIndex][y] = weight[parent]
									* empiricalMEstimate(xyCount[parent][y],
											totalCount,
											xxyDist_.xyCounts.p(y)
													* xxyDist_.xyCounts.p(
															parent, parentVal))
									* scaleFactor;
						}
					} else {
						for (CatValue y = 0; y < noClasses_; y++) {
							spodeProbs[parentIndex][y] = weight[parent]
									* mEstimate(xyCount[parent][y], totalCount,
											noClasses_
													* xxyDist_.getNoValues(
															parent))
									* scaleFactor;
							if (verbosity >= 3) {

								printf("%u,%u,%f\n", parent, y,
										spodeProbs[parent][y]);
							}

						}
					}
				} else if (verbosity >= 5)
					printf("%d\n", parent);
			}
		}

		if (delta == 0) {
			nbClassify(inst, classDist, xxyDist_.xyCounts);
			return;
		}

		for (CatValue parentIndex = 0; parentIndex < optParentIndex_; parentIndex++) {
			CategoricalAttribute parent = orderedAtts[parentIndex];
			const CatValue parentVal = inst.getCatVal(parent);
			if (active[parent] == true) {

				for (CategoricalAttribute childIndex = 0;
						childIndex < optChildIndex_; childIndex++) {

					CategoricalAttribute child;
					if(loo5_==true)
						child= orderedAtts[childIndex];
					else
					{
						assert(loo6_==true);
						child = orderedAttsForParent[parent][childIndex];
					}
					CatValue childVal = inst.getCatVal(child);
					if (child != parent) {
						for (CatValue y = 0; y < noClasses_; y++) {
							spodeProbs[parentIndex][y] *= xxyDist_.p(child,
									childVal, parent, parentVal, y);
						}
					}
				}
			}
		}

		for (CatValue parent = 0; parent < optParentIndex_; parent++) {
			if (active[orderedAtts[parent]]) {
				for (CatValue y = 0; y < noClasses_; y++) {
					classDist[y] += spodeProbs[parent][y];
				}
			}
		}
		normalise(classDist);
		return;
	}


	//Next is for loocv1 and loocv4
	for (CatValue parent = 0; parent < noCatAtts_; parent++) {

		const CatValue parentVal = inst.getCatVal(parent);

		for (CatValue y = 0; y < noClasses_; y++) {
			xyCount[parent][y] = xxyDist_.xyCounts.getCount(parent,
					parentVal, y);
		}

		if (active_[parent]) {
			if (xxyDist_.xyCounts.getCount(parent, parentVal) > 0) {
				delta++;
				active[parent] = true;

				if (empiricalMEst_) {
					for (CatValue y = 0; y < noClasses_; y++) {
						spodeProbs[parent][y] = weight[parent]
								* empiricalMEstimate(xyCount[parent][y],
										totalCount,
										xxyDist_.xyCounts.p(y)
												* xxyDist_.xyCounts.p(
														parent, parentVal))
								* scaleFactor;
					}
				} else {
					for (CatValue y = 0; y < noClasses_; y++) {
						spodeProbs[parent][y] = weight[parent]
								* mEstimate(xyCount[parent][y], totalCount,
										noClasses_
												* xxyDist_.getNoValues(
														parent))
								* scaleFactor;
					}
				}
			} else if (verbosity >= 5)
				printf("%d\n", parent);
		}
	}


	if (delta == 0) {
		nbClassify(inst, classDist, xxyDist_.xyCounts);
		return;
	}

	if(loo1_== true ||loo4_== true)
	{

		for (CatValue parent = 0; parent < noCatAtts_; parent++) {
			if (active[parent] == true) {
				const CatValue parentVal = inst.getCatVal(parent);
				CategoricalAttribute noChild=noCatAtts_;
				if(loo1_==true)
					noChild=noAttSelected_;
//					noChild=1;
				else
				{
					assert(loo4_==true);
					noChild=noChild_[parent];
				}

				for (CategoricalAttribute childIndex = 0; childIndex <noChild ; childIndex++) {
					CategoricalAttribute child ;
					if(loo1_==true)
					{
						child=orderedAtts[childIndex];
					}
					else
					child=orderedAttsForParent[parent][childIndex];

//				for (CategoricalAttribute child = 0; child < noChild; child++) {
					for (CatValue y = 0; y < noClasses_; y++) {
						if (child != parent) {
							CatValue childVal = inst.getCatVal(child);
							spodeProbs[parent][y] *= xxyDist_.p(child, childVal,
									parent, parentVal, y);
						}
					}
				}
			}
		}
	}
	else
	{

		generalizationSet.assign(noCatAtts_, false);
	//	compute the generalisation set and substitution set for
	//	lazy subsumption resolution
		if (subsumptionResolution == true) {
			for (CategoricalAttribute i = 1; i < noCatAtts_; i++) {
				const CatValue iVal = inst.getCatVal(i);
				const InstanceCount countOfxi = xxyDist_.xyCounts.getCount(i, iVal);

				for (CategoricalAttribute j = 0; j < i; j++) {
					if (!generalizationSet[j]) {
						const CatValue jVal = inst.getCatVal(j);
						const InstanceCount countOfxixj = xxyDist_.getCount(i, iVal,
								j, jVal);
						const InstanceCount countOfxj = xxyDist_.xyCounts.getCount(
								j, jVal);

						if (countOfxj == countOfxixj && countOfxj >= minCount) {
							//xi is a generalisation or substitution of xj
							//once one xj has been found for xi, stop for rest j
							generalizationSet[i] = true;
							break;
						} else if (countOfxi == countOfxixj
								&& countOfxi >= minCount) {
							//xj is a generalisation of xi
							generalizationSet[j] = true;
						}
					}
				}
			}

			if (verbosity >= 4) {
				for (CategoricalAttribute i = 0; i < noCatAtts_; i++)
					if (!generalizationSet[i])
						printf("%d\t", i);
				printf("\n");
			}
		}

		if (verbosity >= 4) {
			for (CatValue i = 0; i < noCatAtts_; i++) {
				printf("%f\n", weight[i]);
			}
		}

		for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {

				//discard the attribute that is in generalization set
				if (!generalizationSet[x1]) {
					const CatValue x1Val = inst.getCatVal(x1);
					const unsigned int noX1Vals = xxyDist_.getNoValues(x1);
					const bool x1Active = active[x1];

					constXYSubDist xySubDist(xxyDist_.getXYSubDist(x1, x1Val),
							noClasses_);

					//calculate only for empricial2
					const InstanceCount x1Count = xxyDist_.xyCounts.getCount(x1,
								x1Val);

					for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

						//	printf("c:%d\n", x2);
						if (!generalizationSet[x2]) {
							const bool x2Active = active[x2];

							if (x1Active || x2Active ) {
								CatValue x2Val = inst.getCatVal(x2);
								const unsigned int noX2Vals = xxyDist_.getNoValues(x2);

								    //calculate only for empricial2
									InstanceCount x1x2Count = xySubDist.getCount(
											x2, x2Val, 0);
									for (CatValue y = 1; y < noClasses_; y++) {
										x1x2Count += xySubDist.getCount(x2, x2Val, y);
									}
									const InstanceCount x2Count =
											xxyDist_.xyCounts.getCount(x2, x2Val);

									const double pX2gX1=empiricalMEstimate(x1x2Count,x1Count, xxyDist_.xyCounts.p(x2, x2Val));
									const double pX1gX2=empiricalMEstimate(x1x2Count,x2Count, xxyDist_.xyCounts.p(x1, x1Val));

								for (CatValue y = 0; y < noClasses_; y++) {
									const InstanceCount x1x2yCount = xySubDist.getCount(
											x2, x2Val, y);

									if (x1Active) {
										if (empiricalMEst_) {

											spodeProbs[x1][y] *= empiricalMEstimate(
													x1x2yCount, xyCount[x1][y],
													xxyDist_.xyCounts.p(x2, x2Val));
										} else if (empiricalMEst2_) {
											//double probX2OnX1=mEstimate();
											spodeProbs[x1][y] *= empiricalMEstimate(
													x1x2yCount, xyCount[x1][y],
													pX2gX1);
										} else {
											spodeProbs[x1][y] *= mEstimate(x1x2yCount,
													xyCount[x1][y], noX2Vals);
										}
									}
									if (x2Active ) {
										if (empiricalMEst_) {
											spodeProbs[x2][y] *= empiricalMEstimate(
													x1x2yCount, xyCount[x2][y],
													xxyDist_.xyCounts.p(x1, x1Val));
										} else if (empiricalMEst2_) {
											//double probX2OnX1=mEstimate();
											spodeProbs[x2][y] *= empiricalMEstimate(
													x1x2yCount, xyCount[x2][y],
													pX1gX2);
										} else {
											spodeProbs[x2][y] *= mEstimate(x1x2yCount,
													xyCount[x2][y], noX1Vals);
										}
									}
								}
							}
						}
					}
				}
			}

	}


	for (CatValue parent = 0; parent < noCatAtts_; parent++) {
		if (active[parent]) {
			for (CatValue y = 0; y < noClasses_; y++) {
				classDist[y] += spodeProbs[parent][y];
			}
		}
	}

	normalise(classDist);
}


void aodeChen::nbClassify(const instance &inst, std::vector<double> &classDist,
		xyDist &xyDist_) {

	for (CatValue y = 0; y < noClasses_; y++) {
		double p = xyDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);
		// scale up by maximum possible factor to reduce risk of numeric underflow

		for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
			p *= xyDist_.p(a, inst.getCatVal(a), y);
		}

		assert(p >= 0.0);
		classDist[y] = p;
	}
	normalise(classDist);
}

