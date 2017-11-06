/*
 * a2de3.cpp
 *
 *  Created on: 28/09/2012
 *      Author: shengleichen
 */

#include "a2de3.h"
#include "assert.h"
#include "utils.h"
#include <algorithm>
#include "correlationMeasures.h"
#include "globals.h"
#include "instanceStream.h"

a2de3::a2de3(char* const *& argv, char* const * end) :
		pass_(1), weight(1) {
	name_ = "a2de3";

	// TODO Auto-generated constructor stub
//	weighted = false;
//	minCount = 100;
//	subsumptionResolution = false;
//	avg = false;
	selected = false;

	acmi_ = false;

	avg_=false;
	sum_=false;


	su_ = false;
	mi_ = false;
	empiricalMEst_ = false;
	empiricalMEst2_ = false;
	memorySelective_ = false;

	factor_ = 1;  //default value
//
// get arguments
	while (argv != end) {
		if (*argv[0] != '-') {
			break;
		} else if (streq(argv[0] + 1, "empirical")) {
			empiricalMEst_ = true;
		} else if (streq(argv[0] + 1, "empirical2")) {
			empiricalMEst2_ = true;
//		} else if (streq(argv[0] + 1, "w")) {
//			weighted = true;
//		} else if (streq(argv[0] + 1, "sub")) {
//			subsumptionResolution = true;
//		} else if (argv[0][1] == 'c') {
//			getUIntFromStr(argv[0] + 2, minCount, "c");
//		} else if (streq(argv[0] + 1, "avg")) {
//			avg = true;
		} else if (streq(argv[0] + 1, "selective")) {
			selected = true;
		} else if (streq(argv[0] + 1, "acmi")) {
			selected = true;
			acmi_ = true;
		} else if (streq(argv[0] + 1, "sum")) {
			sum_ = true;
		} else if (streq(argv[0] + 1, "avg")) {
			avg_ = true;
		} else if (streq(argv[0] + 1, "mi")) {
			selected = true;
			mi_ = true;
		} else if (argv[0][1] == 'f') {
			unsigned int factor;
			getUIntFromStr(argv[0] + 2, factor, "f");
			factor_ = factor / 10.0;
			while (factor_ >= 1)
				factor_ /= 10;
		} else if (argv[0][1] == 'o') {
			memorySelective_ = true;
		} else if (streq(argv[0] + 1, "su")) {
			selected = true;
			su_ = true;
		} else if (streq(argv[0] + 1, "chisq")) {
			selected = true;
			chisq_ = true;
		} else {
			error("a2de3 does not support argument %s\n", argv[0]);
			break;
		}

		name_ += *argv;

		++argv;
	}

	if (selected == true) {
		if (mi_ == false && su_ == false && chisq_ == false)
			chisq_ = true;
	}

	if(sum_==false&& avg_==false)
		sum_=true;

	trainingIsFinished_ = false;
}

a2de3::~a2de3() {
	// TODO Auto-generated destructor stub
}

void a2de3::reset(InstanceStream &is) {
	xxyDist_.reset(is);

	trainingIsFinished_ = false;

	noCatAtts_ = is.getNoCatAtts();
	noClasses_ = is.getNoClasses();

	inactiveCnt_ = 0;

	if (memorySelective_ == false)
	     noSelectedCatAtts_ = static_cast<unsigned int>(noCatAtts_ * factor_); ///< the number of selected attributes
	else {
		unsigned int v;
		unsigned int i;
		v = is.getNoValues(0);
		for (i = 1; i < noCatAtts_; i++)
			v += is.getNoValues(i);
		v = v / noCatAtts_;


		/// memory of selective A2DE equal to memory of AODE
		///  k*a^2*v^2=k*s^3*v^3
		///  where k: number of classes
		///        a: number of attributes
		///        v: average number of values for each attribute
		///        s: number of selected attributes

		/// simplification:  a^2=s^3*v

		for (noSelectedCatAtts_ = noCatAtts_;
				noCatAtts_ * noCatAtts_
						< noSelectedCatAtts_ * noSelectedCatAtts_
								* noSelectedCatAtts_ * v; noSelectedCatAtts_--)
			;

		//guarantee to select at least one attribute
		if(noSelectedCatAtts_==0)
			noSelectedCatAtts_=1;

		printf("The number of attributes and selected attributes: %u,%u\n",
				noCatAtts_, noSelectedCatAtts_);
		printf("The average number of values for all attributes: %u\n", v);

	}

	noUnSelectedCatAtts_ = noCatAtts_ - noSelectedCatAtts_;
	order_.clear();

	weight = crosstab<double>(noCatAtts_);

	weightaode.assign(noCatAtts_, 1);

	//initialise the weight for non-weighting

	for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
		for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

			weight[x1][x2] = 1;
			weight[x2][x1] = 1;
		}
	}

	generalizationSet.assign(noCatAtts_, false);

	active_.assign(noCatAtts_, true);
	instanceStream_ = &is;
	pass_ = 1;
	count = 0;

	xxxyDist_.setNoSelectedCatAtts(noSelectedCatAtts_);
	//xxxyDist_.reset(is);
}

void a2de3::getCapabilities(capabilities &c) {
	c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void a2de3::initialisePass() {

}

/// true iff no more passes are required. updated by finalisePass()
bool a2de3::trainingIsFinished() {
	return pass_ > 2;
}

void a2de3::train(const instance &inst) {

	if (pass_ == 1)
		xxyDist_.update(inst);
	else {
		assert(pass_ == 2);
		xxxyDist_.update(inst);
	}
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

void a2de3::finalisePass() {
	if (pass_ == 1) {

		for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
			order_.push_back(a);
		}

		if (mi_ == true || su_ == true || acmi_ == true) {
			//calculate the symmetrical uncertainty between each attribute and class
			std::vector<float> measure;
			crosstab<float> acmi(noCatAtts_);

			if (mi_ == true && acmi_ == true) {
				if (sum_ == true) {
					getAttClassCondMutualInf(xxyDist_, acmi);
					getMutualInformation(xxyDist_.xyCounts, measure);

					for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
						for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

							measure[x1] += acmi[x2][x1];
							measure[x2] += acmi[x1][x2];
							if (verbosity >= 5) {
								if (x1 == 2)
									printf("%u,", x2);
								if (x2 == 2)
									printf("%u,", x1);
							}
						}
					}

				} else if (avg_ == true) {
					getAttClassCondMutualInf(xxyDist_, acmi);
					measure.assign(noCatAtts_, 0.0);
					for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
						for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

							measure[x1] += acmi[x2][x1];
							measure[x2] += acmi[x1][x2];
							if (verbosity >= 5) {
								if (x1 == 2)
									printf("%u,", x2);
								if (x2 == 2)
									printf("%u,", x1);
							}
						}
					}
					std::vector<float> mi;
					getMutualInformation(xxyDist_.xyCounts, mi);

					for (CategoricalAttribute x1 = 0; x1 < noCatAtts_; x1++) {
						measure[x1] = measure[x1] / (noCatAtts_ - 1) + mi[x1];
					}
				}
			} else if (mi_ == true)
				getMutualInformation(xxyDist_.xyCounts, measure);
			else if (su_ == true)
				getSymmetricalUncert(xxyDist_.xyCounts, measure);
			else if (acmi_ == true) {
				getAttClassCondMutualInf(xxyDist_, acmi);
				measure.assign(noCatAtts_, 0.0);

				for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
					for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

						measure[x1] += acmi[x2][x1];
						measure[x2] += acmi[x1][x2];
						if (verbosity >= 5) {
							if (x1 == 2)
								printf("%u,", x2);
							if (x2 == 2)
								printf("%u,", x1);
						}
					}
				}
			}

			if (verbosity >= 2) {
				if (mi_ == true && acmi_ == true) {
					if (sum_ == true)
						printf(
								"Selecting according to mutual information and sum acmi:\n");
					else if (avg_ == true)
						printf(
								"Selecting according to mutual information and avg acmi:\n");
				} else if (mi_ == true)
					printf("Selecting according to mutual information:\n");
				else if (su_ == true)
					printf("Selecting according to symmetrical uncertainty:\n");
				else if (acmi_ == true)
					printf(
							"Selecting according to attribute and class conditional mutual information:\n");

				print(measure);
				printf("\n");
			}

			if (!order_.empty()) {

				valCmpClass cmp(&measure);
				std::sort(order_.begin(), order_.end(), cmp);

				if (verbosity >= 2) {
					const char * sep = "";
					printf("The order of attributes ordered by the measure:\n");
					for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
						printf("%d:\t%f\n", order_[a], measure[order_[a]]);
						sep = ", ";
					}
					printf("\n");
				}

				//order by attribute number for selected attributes
				std::sort(order_.begin(), order_.begin() + noSelectedCatAtts_);

				//set the attribute selected or unselected for aode
				for (CategoricalAttribute a = noSelectedCatAtts_;
						a < noCatAtts_; a++) {
					active_[order_[a]] = false;
				}

				if (verbosity >= 2) {
					const char * sep = "";
					if (memorySelective_ == true)
						printf(
								"The attributes being selected according to the memory:\n");
					else
						printf("The attributes specified by the user:\n");
					for (CategoricalAttribute a = 0; a < noSelectedCatAtts_;
							a++) {
						printf("%s%d", sep, order_[a]);
						sep = ", ";
					}
					printf("\n");
				}

			}

		}
		xxxyDist_.setOrder(order_);
		xxxyDist_.reset(*instanceStream_);
	}
	pass_++;
}

void a2de3::classify(const instance &inst, std::vector<double> &classDist) {
	if (verbosity >= 2)
		count++;

	unsigned int check = 101;

	if (verbosity == 3 && count == check) {
		printf("current instance:\n");
		for (CategoricalAttribute x1 = 0; x1 < noCatAtts_; x1++) {
			printf("%u:%u\n", x1, inst.getCatVal(x1));
		}
	}

	generalizationSet.assign(noCatAtts_, false);

	xxyDist * xxydist = &xxxyDist_.xxyCounts;
	xyDist * xydist = &xxxyDist_.xxyCounts.xyCounts;

	const InstanceCount totalCount = xydist->count;


	for (CatValue y = 0; y < noClasses_; y++)
		classDist[y] = 0;

	CatValue delta = 0;

////	 scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max()
			/ ((noSelectedCatAtts_) * (noSelectedCatAtts_ - 1) / 2.0);

//	 scale up by maximum possible factor to reduce risk of numeric underflow
//	double scaleFactor = 1;

//	// first to assign the spodeProbs array
	std::vector<std::vector<std::vector<double> > > spodeProbs;
	spodeProbs.resize(noSelectedCatAtts_);

	for (CatValue father = 1; father < noSelectedCatAtts_; father++) {
		spodeProbs[father].resize(father);
		for (CatValue mother = 0; mother < father; mother++)
			spodeProbs[father][mother].assign(noClasses_, 0);
	}

	//crosstab<bool> activeParent(noSelectedCatAtts_);

	for (CatValue father = 1; father < noSelectedCatAtts_; father++) {

		//select the attribute according to centain measure
//		if (active_[father] == false)
//			continue;
		//selecct attribute for subsumption resolution
		if (generalizationSet[father])
			continue;
		const CatValue fatherVal = inst.getCatVal(order_[father]);

		for (CatValue mother = 0; mother < father; mother++) {
			//select the attribute according to centain measure
//			if (active_[mother] == false)
//				continue;
			//selecct attribute for subsumption resolution
			if (generalizationSet[mother])
				continue;
			const CatValue motherVal = inst.getCatVal(order_[mother]);

			CatValue parent = 0;
			for (CatValue y = 0; y < noClasses_; y++) {
				parent += xxydist->getCount(order_[father], fatherVal,
						order_[mother], motherVal, y);
			}

			if (parent > 0) {
			//	activeParent[father][mother] = true;

				delta++;

				for (CatValue y = 0; y < noClasses_; y++) {
					spodeProbs[father][mother][y] = weight[father][mother]
							* scaleFactor; // scale up by maximum possible factor to reduce risk of numeric underflow

					InstanceCount parentYCount = xxydist->getCount(
							order_[father], fatherVal, order_[mother],
							motherVal, y);
					if (empiricalMEst_) {

						spodeProbs[father][mother][y] *= empiricalMEstimate(
								parentYCount, totalCount,
								xydist->p(y)
										* xydist->p(order_[father], fatherVal)
										* xydist->p(order_[mother], motherVal));

					} else {
						double temp = mEstimate(parentYCount, totalCount,
								noClasses_ * xxxyDist_.getNoValues(father)
										* xxxyDist_.getNoValues(mother));
						spodeProbs[father][mother][y] *= temp;
					}

					if (verbosity == 4)
						printf("%f,", spodeProbs[father][mother][y]);
				}

			}
		}

	}

	if (verbosity == 3 & count == check) {
		for (CatValue father = 1; father < noSelectedCatAtts_; father++) {
			for (CatValue mother = 0; mother < father; mother++) {
				printf("initial spode probs for %u,%u:\n", order_[father],
						order_[mother]);
				print(spodeProbs[father][mother]);
				printf("\n");
			}
		}
		printf("initial spode probs ending.<<<<<\n");
	}

	//count++;
	if (delta == 0) {

		aodeClassify(inst, classDist, *xxydist);

		return;
	}

	if (verbosity == 3 && count == check) {
		printf("print every prob for each parents:\n");
	}

	for (CatValue father = 1; father < noSelectedCatAtts_; father++) {

		//select the attribute according to centain measure
//		if (active_[father] == false)
//			continue;
		//selecct attribute for subsumption resolution
		if (generalizationSet[father])
			continue;
		const CatValue fatherVal = inst.getCatVal(order_[father]);
//
		XXYSubDist xxySubDist(xxxyDist_.getXXYSubDist(father, fatherVal),
				noClasses_);

		XXYSubDist xxySubDistRest(
				xxxyDist_.getXXYSubDistRest(father, fatherVal), noClasses_);

		XYSubDist xySubDistFather(
				xxydist->getXYSubDist(order_[father], fatherVal), noClasses_);

		for (CatValue mother = 0; mother < father; mother++) {
			//select the attribute according to centain measure
			if (verbosity == 3 && order_[father] == 6 && order_[mother] == 0
					&& count == check)
				printf("%u\n", check);

			//selecct attribute for subsumption resolution
			if (generalizationSet[mother])
				continue;
			const CatValue motherVal = inst.getCatVal(order_[mother]);

			XYSubDist xySubDist(xxySubDist.getXYSubDist(mother, motherVal),
					noClasses_);

			XYSubDist xySubDistRest(
					xxySubDistRest.getXYSubDist(mother, motherVal,
							noUnSelectedCatAtts_), noClasses_);

			XYSubDist xySubDistMother(
					xxydist->getXYSubDist(order_[mother], motherVal),
					noClasses_);

			if (verbosity == 3 && count == check) {
				printf("first selected attributes as child\n");
			}
			// as we store the instance count completely for the third attributes
			// we here can set child to all possible values
			for (CatValue child = 0; child < mother; child++) {

				if (!generalizationSet[child]) {

					const CatValue childVal = inst.getCatVal(order_[child]);

					if (child != father && child != mother) {
						//if (order_[child] != order_[father] && order_[child] != order_[mother]) {
//							if (verbosity == 3 && count == 1)
//								printf("why no output \n");
						for (CatValue y = 0; y < noClasses_; y++) {

							InstanceCount parentChildYCount =
									xySubDist.getCount(child, childVal, y);

							InstanceCount parentYCount =
									xySubDistFather.getCount(order_[mother],
											motherVal, y);
							InstanceCount fatherYChildCount =
									xySubDistFather.getCount(order_[child],
											childVal, y);
							InstanceCount motherYChildCount =
									xySubDistMother.getCount(order_[child],
											childVal, y);

							double temp;
							if (empiricalMEst_) {
								if (xxydist->getCount(order_[father], fatherVal,
										order_[mother], motherVal) > 0) {

									temp = empiricalMEstimate(parentChildYCount,
											parentYCount,
											xydist->p(order_[child], childVal));
									spodeProbs[father][mother][y] *= temp;
								}
								if (xxydist->getCount(order_[father], fatherVal,
										order_[child], childVal) > 0) {
									temp = empiricalMEstimate(parentChildYCount,
											fatherYChildCount,
											xydist->p(mother, motherVal));
									spodeProbs[father][child][y] *= temp;
								}

								if (xxydist->getCount(order_[mother], motherVal,
										order_[child], childVal) > 0) {
									temp = empiricalMEstimate(parentChildYCount,
											motherYChildCount,
											xydist->p(father, fatherVal));
									spodeProbs[mother][child][y] *= temp;

								}

							} else {

								if (xxydist->getCount(order_[father], fatherVal,
										order_[mother], motherVal) > 0) {
									temp = mEstimate(parentChildYCount,
											parentYCount,
											xxxyDist_.getNoValues(child));
									spodeProbs[father][mother][y] *= temp;

									if (verbosity == 3 && count == check
											&& y == 0) {
										printf("%u,%u,%u,%u,%f\n",
												order_[father], order_[mother],
												order_[child], y, temp);
									}

								}
								if (xxydist->getCount(order_[father], fatherVal,
										order_[child], childVal) > 0) {

									temp = mEstimate(parentChildYCount,
											fatherYChildCount,
											xxxyDist_.getNoValues(mother));
									spodeProbs[father][child][y] *= temp;

									if (verbosity == 3 && count == check
											&& y == 0) {
										printf("%u,%u,%u,%u,%f\n",
												order_[father], order_[child],
												order_[child], y, temp);
									}
								}
								if (xxydist->getCount(order_[mother], motherVal,
										order_[child], childVal) > 0) {

									temp = mEstimate(parentChildYCount,
											motherYChildCount,
											xxxyDist_.getNoValues(father));
									spodeProbs[mother][child][y] *= temp;

									if (verbosity == 3 && count == check
											&& y == 0) {
										printf("%u,%u,%u,%u,%f---\n",
												order_[mother], order_[child],
												order_[child], y, temp);
									}
								}
								if (verbosity == 3 && count == check
										&& y == 0) {
									if (order_[father] == 3
											&& order_[mother] == 1
											&& order_[child] == 5) {
										printf("%u,%u,%u\n", parentChildYCount,
												parentYCount,
												xxxyDist_.getNoValues(child));
									}

								}

								if (verbosity == 3 && count == check
										&& y == 0) {
									if (order_[father] == 3
											&& order_[child] == 1
											&& order_[mother] == 5) {
										printf("%u,%u,%u\n", parentChildYCount,
												fatherYChildCount,
												xxxyDist_.getNoValues(mother));
									}

								}

								if (verbosity == 3 && count == check
										&& y == 0) {
									if (order_[mother] == 3
											&& order_[child] == 1
											&& order_[father] == 5) {
										printf("%u,%u,%u---\n",
												parentChildYCount,
												motherYChildCount,
												xxxyDist_.getNoValues(father));
									}

								}

							}

						}
					}
				}
			}
			if (verbosity == 3 && count == check) {
				printf("next unselected attributes as child\n");
			}

			//check if the parent is qualified
			if (xxydist->getCount(order_[father], fatherVal, order_[mother],
					motherVal) == 0)
				continue;

			for (CatValue child = 0; child < noUnSelectedCatAtts_; child++) {
				if (verbosity == 3 && order_[child + noSelectedCatAtts_] == 13
						&& count == check)
					printf("%u\n", check);
				if (!generalizationSet[child]) {

					const CatValue childVal = inst.getCatVal(
							order_[child + noSelectedCatAtts_]);

					//if (child != father && child != mother) {
					//if (order_[child] != order_[father] && order_[child] != order_[mother]) {

					for (CatValue y = 0; y < noClasses_; y++) {

						InstanceCount parentChildYCount =
								xySubDistRest.getCount(child, childVal, y);

						InstanceCount parentYCount = xySubDistFather.getCount(
								order_[mother], motherVal, y);

						double temp;
						if (empiricalMEst_) {

							temp = empiricalMEstimate(parentChildYCount,
									parentYCount,
									xydist->p(order_[child], childVal));
							spodeProbs[father][mother][y] *= temp;

						} else {

							temp = mEstimate(parentChildYCount, parentYCount,
									xxxyDist_.getNoValues(
											child + noSelectedCatAtts_));
							spodeProbs[father][mother][y] *= temp;

							if (verbosity == 3 && count == check && y == 0) {
								if (order_[father] == 3 && order_[mother] == 1
										&& order_[child + noSelectedCatAtts_]
												== 5) {
									printf("%u,%u,%u\n", parentChildYCount,
											parentYCount,
											xxxyDist_.getNoValues(
													child
															+ noSelectedCatAtts_));
								}

							}
							if (verbosity == 3 && count == check && y == 0) {
								printf("%u,%u,%u,%u,%f\n", order_[father],
										order_[mother],
										order_[child + noSelectedCatAtts_], y,
										temp);
							}

						}
					}
				}

			}

		}

	}

	for (CatValue father = 1; father < noSelectedCatAtts_; father++) {
		for (CatValue mother = 0; mother < father; mother++)
			for (CatValue y = 0; y < noClasses_; y++) {
				classDist[y] += spodeProbs[father][mother][y];
			}
	}

	if (verbosity == 3 && count == check) {
		printf("distribute of instance %u for each parent.\n", count);
		for (CatValue father = 1; father < noSelectedCatAtts_; father++) {
			for (CatValue mother = 0; mother < father; mother++) {
				printf("%u,%u:", order_[father], order_[mother]);
				print(spodeProbs[father][mother]);
				printf("\n");
			}
		}
		printf("spode probs ending.<<<<<\n");
	}

	if (verbosity == 4) {
		if (count == check) {
			printf("the class dist of instance %u before normalizing:\n",
					count);
			for (unsigned int i = 0; i < classDist.size(); i++)
				printf("%0.14f,", classDist[i]);
			printf("\n");
		}
	}

	normalise(classDist);

	if (verbosity >= 3) {
		if (count == check) {
			printf("the class dist of instance %u:\n", count);
			for (unsigned int i = 0; i < classDist.size(); i++)
				printf("%0.14f,", classDist[i]);
			printf("\n");
		}
	}

}

void a2de3::aodeClassify(const instance &inst, std::vector<double> &classDist,
		xxyDist & xxyDist_) {

// scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max() / noCatAtts_;

	CatValue delta = 0;

	const InstanceCount totalCount = xxyDist_.xyCounts.count;

	fdarray<double> spodeProbs(noCatAtts_, noClasses_);
	fdarray<InstanceCount> xyCount(noCatAtts_, noClasses_);
	std::vector<bool> active(noCatAtts_, false);

	for (CatValue parent = 0; parent < noCatAtts_; parent++) {

		//discard the attribute that is not active or in generalization set
		if (!generalizationSet[parent]) {
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
							spodeProbs[parent][y] = weightaode[parent]
									* empiricalMEstimate(xyCount[parent][y],
											totalCount,
											xxyDist_.xyCounts.p(y)
													* xxyDist_.xyCounts.p(
															parent, parentVal))
									* scaleFactor;
						}
					} else {
						for (CatValue y = 0; y < noClasses_; y++) {
							spodeProbs[parent][y] = weightaode[parent]
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
	}

	if (delta == 0) {
		count++;
		if (verbosity == 2)
			printf("aode is called for %u times\n", count);
		nbClassify(inst, classDist, xxyDist_.xyCounts);
		return;
	}

	for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
		//
		//		std::vector<std::vector<std::vector<double> > > * parentsProbs =
		//				&xxyDist_.condiProbs[x1][x1Val];

		//discard the attribute that is in generalization set
		if (!generalizationSet[x1]) {
			const CatValue x1Val = inst.getCatVal(x1);
			const unsigned int noX1Vals = xxyDist_.getNoValues(x1);
			const bool x1Active = active[x1];

			constXYSubDist xySubDist(xxyDist_.getXYSubDist(x1, x1Val),
					noClasses_);

			//calculate only for empricial2
			const InstanceCount x1Count = xxyDist_.xyCounts.getCount(x1, x1Val);

			for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

				//	printf("c:%d\n", x2);
				if (!generalizationSet[x2]) {
					const bool x2Active = active[x2];

					if (x1Active || x2Active) {
						CatValue x2Val = inst.getCatVal(x2);
						const unsigned int noX2Vals = xxyDist_.getNoValues(x2);

						//calculate only for empricial2
						InstanceCount x1x2Count = xySubDist.getCount(x2, x2Val,
								0);
						for (CatValue y = 1; y < noClasses_; y++) {
							x1x2Count += xySubDist.getCount(x2, x2Val, y);
						}
						const InstanceCount x2Count =
								xxyDist_.xyCounts.getCount(x2, x2Val);

						const double pX2gX1 = empiricalMEstimate(x1x2Count,
								x1Count, xxyDist_.xyCounts.p(x2, x2Val));
						const double pX1gX2 = empiricalMEstimate(x1x2Count,
								x2Count, xxyDist_.xyCounts.p(x1, x1Val));

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
											x1x2yCount, xyCount[x1][y], pX2gX1);
								} else {
									spodeProbs[x1][y] *= mEstimate(x1x2yCount,
											xyCount[x1][y], noX2Vals);
								}
							}
							if (x2Active) {
								if (empiricalMEst_) {
									spodeProbs[x2][y] *= empiricalMEstimate(
											x1x2yCount, xyCount[x2][y],
											xxyDist_.xyCounts.p(x1, x1Val));
								} else if (empiricalMEst2_) {
									//double probX2OnX1=mEstimate();
									spodeProbs[x1][y] *= empiricalMEstimate(
											x1x2yCount, xyCount[x1][y], pX1gX2);
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

	for (CatValue parent = 0; parent < noCatAtts_; parent++) {
		if (active[parent]) {
			for (CatValue y = 0; y < noClasses_; y++) {
				classDist[y] += spodeProbs[parent][y];
			}
		}
	}

	normalise(classDist);

	count++;
	if (verbosity == 2 && count == 1) {
		printf("the class distribution is :\n");
		print(classDist);
		printf("\n");

	}
}
void a2de3::nbClassify(const instance &inst, std::vector<double> &classDist,
		xyDist & xyDist_) {

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

