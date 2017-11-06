/*
 * a3de3.cpp
 *
 *  Created on: 28/09/2012
 *      Author: shengleichen
 */

#include "a3de3.h"
#include "assert.h"
#include "utils.h"
#include <algorithm>
#include "correlationMeasures.h"

#include "globals.h"
#include "instanceStream.h"

a3de3::a3de3(char* const *& argv, char* const * end) :
		weight_a2de(1) {
	name_ = "A3DE3";

	// TODO Auto-generated constructor stub
	count_=0;
	weighted = false;
	minCount = 100;
	subsumptionResolution = false;
	selected = false;

//	chisq_ = false;
	acmi_ = false;
	mi_ = false;
	su_ = false;
	sum_ = false;
	avg_ = false;

	oneSelective_ = false;
	twoSelective_ = false;

	memorySelective_ = false;

	empiricalMEst_ = false;
	empiricalMEst2_ = false;

	factor_ = 1;  //default value

	// get arguments
	while (argv != end) {
		if (*argv[0] != '-') {
			break;
		} else if (streq(argv[0] + 1, "empirical")) {
			empiricalMEst_ = true;
		} else if (streq(argv[0] + 1, "empirical2")) {
			empiricalMEst2_ = true;
		} else if (streq(argv[0] + 1, "selective")) {
			selected = true;

//		} else if (streq(argv[0] + 1, "chisq")) {
//			selected = true;
//			chisq_ = true;
		} else if (streq(argv[0] + 1, "acmi")) {
			selected = true;
			acmi_ = true;
		} else if (streq(argv[0] + 1, "mi")) {
			selected = true;
			mi_ = true;
		} else if (streq(argv[0] + 1, "su")) {
			selected = true;
			su_ = true;


		} else if (streq(argv[0] + 1, "sum")) {
			sum_ = true;
		} else if (streq(argv[0] + 1, "avg")) {
			avg_ = true;

		} else if (argv[0][1] == 'f') {
			unsigned int factor;
			getUIntFromStr(argv[0] + 2, factor, "f");
			factor_ = factor / 10.0;
			while (factor_ >= 1)
				factor_ /= 10;
		} else if (streq(argv[0] + 1, "one")) {
			oneSelective_ = true;
		} else if (streq(argv[0] + 1, "two")) {
			twoSelective_ = true;

		} else {
			error("a3de3 does not support argument %s\n", argv[0]);
			break;
		}

		name_ += *argv;

		++argv;
	}

	if (selected == true) {
		if (mi_ == false && su_ == false && chisq_ == false)
			mi_ = true;
	}

	if (sum_ == false && avg_ == false)
		sum_ = true;

	trainingIsFinished_ = false;
	printf("Classifier %s is constructed.\n",name_.c_str());
}

a3de3::~a3de3() {
	// TODO Auto-generated destructor stub
}

void a3de3::reset(InstanceStream &is) {


	xxyDist_.reset(is);
	instanceStream_ = &is;
	trainingIsFinished_ = false;

	noCatAtts_ = is.getNoCatAtts();
	noClasses_ = is.getNoClasses();
	inactiveCnt_ = 0;

	unsigned int v;
	unsigned int i;
	v = is.getNoValues(0);
	for (i = 1; i < noCatAtts_; i++)
		v += is.getNoValues(i);
	v = v / noCatAtts_;

	if (oneSelective_) {

		for (noSelectedCatAtts_ = noCatAtts_;
				noCatAtts_ * noCatAtts_
						< noSelectedCatAtts_ * noSelectedCatAtts_
								* noSelectedCatAtts_ * noSelectedCatAtts_ * v
								* v; noSelectedCatAtts_--)
			;
		printf("Select the attributes according to the memory aode required.\n");
		printf("The number of attributes and selected attributes: %u,%u\n",
				noCatAtts_, noSelectedCatAtts_);
		printf("The average number of values for all attributes: %u\n", v);

	} else if (twoSelective_) {
		for (noSelectedCatAtts_ = noCatAtts_;
				noCatAtts_ * noCatAtts_ * noCatAtts_
						< noSelectedCatAtts_ * noSelectedCatAtts_
								* noSelectedCatAtts_ * noSelectedCatAtts_ * v;
				noSelectedCatAtts_--)
			;
		printf("Selecting the attributes according to the memory a2de required.\n");
		printf("The number of attributes and selected attributes: %u,%u\n",
				noCatAtts_, noSelectedCatAtts_);
		printf("The average number of values for all attributes: %u\n", v);

	} else

		noSelectedCatAtts_ = static_cast<unsigned int>(noCatAtts_ * factor_); ///< the number of selected attributes


	noUnSelectedCatAtts_ = noCatAtts_ - noSelectedCatAtts_;
	order_.clear();

	weight_aode.assign(noCatAtts_, 1);
	weight_a2de = crosstab<double>(noCatAtts_);
	weight = crosstab3D<double>(noCatAtts_);

	//initialise the weight for non-weighting
	for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
		for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

			weight_a2de[x1][x2] = 1;
			weight_a2de[x2][x1] = 1;
		}
	}

	for (CategoricalAttribute x1 = 2; x1 < noCatAtts_; x1++) {
		for (CategoricalAttribute x2 = 1; x2 < x1; x2++) {
			for (CategoricalAttribute x3 = 0; x3 < x2; x3++) {
				weight[x1][x2][x3] = 1;
				weight[x2][x1][x3] = 1;
				weight[x1][x3][x2] = 1;
				weight[x2][x3][x1] = 1;
				weight[x3][x1][x2] = 1;
				weight[x3][x2][x1] = 1;
			}
		}
	}

	generalizationSet.assign(noCatAtts_, false);

	active_.assign(noCatAtts_, true);

	pass_ = 1;
	//count = 0;

	xxxxyDist_.setNoSelectedCatAtts(noSelectedCatAtts_);
	//xxxxyDist_.reset(is);
}

void a3de3::getCapabilities(capabilities &c) {
	c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void a3de3::initialisePass() {

}

/// true iff no more passes are required. updated by finalisePass()
bool a3de3::trainingIsFinished() {
	return pass_ > 2;
}

void a3de3::train(const instance &inst) {

	if (pass_ == 1)
		xxyDist_.update(inst);
	else {
		assert(pass_ == 2);
		xxxxyDist_.update(inst);
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

void a3de3::finalisePass() {

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
					if (oneSelective_ == true)
						printf(
								"The attributes being selected according to the memory of AODE:\n");
					else if (twoSelective_ == true)
											printf(
													"The attributes being selected according to the memory of A2DE:\n");
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
		xxxxyDist_.setOrder(order_);
		xxxxyDist_.reset(*instanceStream_);

	}

	pass_++;
}

void a3de3::classify(const instance &inst, std::vector<double> &classDist) {
	count_++;
	generalizationSet.assign(noCatAtts_, false);

	xxxyDist * xxxydist = &xxxxyDist_.xxxyCounts;
	xxyDist * xxydist = &xxxydist->xxyCounts;
	xyDist * xydist = &xxydist->xyCounts;

	const InstanceCount totalCount = xydist->count;

	for (CatValue y = 0; y < noClasses_; y++)
		classDist[y] = 0;

	CatValue delta = 0;

	// scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max()
			/ ((noSelectedCatAtts_ - 1) * (noSelectedCatAtts_ - 2)
					* (noSelectedCatAtts_) / 6.0);

	//	// first to assign the spodeProbs array
	std::vector<std::vector<std::vector<std::vector<double> > > > spodeProbs;
	spodeProbs.resize(noSelectedCatAtts_);

	for (CatValue parent1 = 2; parent1 < noSelectedCatAtts_; parent1++) {
		spodeProbs[parent1].resize(parent1);
		for (CatValue parent2 = 1; parent2 < parent1; parent2++) {
			spodeProbs[parent1][parent2].resize(parent2);
			for (CatValue parent3 = 0; parent3 < parent2; parent3++)
				spodeProbs[parent1][parent2][parent3].assign(noClasses_, 0);
		}
	}

	for (CatValue parent1 = 2; parent1 < noSelectedCatAtts_; parent1++) {

		//selecct attribute for subsumption resolution
		if (generalizationSet[parent1])
			continue;
		const CatValue parent1Val = inst.getCatVal(order_[parent1]);

		for (CatValue parent2 = 1; parent2 < parent1; parent2++) {

			//selecct attribute for subsumption resolution
			if (generalizationSet[parent2])
				continue;
			const CatValue parent2Val = inst.getCatVal(order_[parent2]);

			for (CatValue parent3 = 0; parent3 < parent2; parent3++) {

				//selecct attribute for subsumption resolution
				if (generalizationSet[parent3])
					continue;
				const CatValue parent3Val = inst.getCatVal(order_[parent3]);

				CatValue parent = 0;
				for (CatValue y = 0; y < noClasses_; y++) {
					parent += xxxydist->getCount(order_[parent1], parent1Val,
							order_[parent2], parent2Val, order_[parent3],
							parent3Val, y);
				}

				if (parent > 0) {

					delta++;

					for (CatValue y = 0; y < noClasses_; y++) {
						spodeProbs[parent1][parent2][parent3][y] =
								weight[parent1][parent2][parent3] * scaleFactor; // scale up by maximum possible factor to reduce risk of numeric underflow

						InstanceCount parentYCount = xxxydist->getCount(
								order_[parent1], parent1Val, order_[parent2],
								parent2Val, order_[parent3], parent3Val, y);
						if (empiricalMEst_) {

							spodeProbs[parent1][parent2][parent3][y] *=
									empiricalMEstimate(parentYCount, totalCount,
											xydist->p(y)
													* xydist->p(order_[parent1],
															parent1Val)
													* xydist->p(order_[parent2],
															parent2Val)
													* xydist->p(order_[parent3],
															parent3Val));

						} else {
							double temp = mEstimate(parentYCount, totalCount,
									noClasses_ * xxxxyDist_.getNoValues(parent1)
											* xxxxyDist_.getNoValues(parent2)
											* xxxxyDist_.getNoValues(parent3));
							spodeProbs[parent1][parent2][parent3][y] *= temp;
						}
					}

				}
			}

		}
	}

	if (delta == 0) {

		a2deClassify(inst, classDist, *xxxydist);

		return;
	}

	//deal with the selected attributes as parents and child
	for (CatValue parent1 = 3; parent1 < noSelectedCatAtts_; parent1++) {
		if (generalizationSet[parent1])
			continue;
		const CatValue parent1Val = inst.getCatVal(order_[parent1]);

		for (CatValue parent2 = 2; parent2 < parent1; parent2++) {
			if (generalizationSet[parent2])
				continue;
			const CatValue parent2Val = inst.getCatVal(order_[parent2]);

			XXYSubDist xxySubDist(
					xxxxyDist_.getXXYSubDist(parent1, parent1Val, parent2,
							parent2Val), noClasses_);

			XYSubDist xySubDistParent12(
					xxxydist->getXYSubDist(order_[parent1], parent1Val,
							order_[parent2], parent2Val), noClasses_);

			for (CatValue parent3 = 1; parent3 < parent2; parent3++) {
				if (generalizationSet[parent3])
					continue;
				const CatValue parent3Val = inst.getCatVal(order_[parent3]);

				XYSubDist xySubDist(
						xxySubDist.getXYSubDist(parent3, parent3Val),
						noClasses_);

				XYSubDist xySubDistParent23(
						xxxydist->getXYSubDist(order_[parent2], parent2Val,
								order_[parent3], parent3Val), noClasses_);
				XYSubDist xySubDistParent13(
						xxxydist->getXYSubDist(order_[parent1], parent1Val,
								order_[parent3], parent3Val), noClasses_);

				for (CatValue child = 0; child < parent3; child++) {

					if(verbosity>=2&&count_==1)
					{
						printf("%u,%u,%u>>%u\n",parent1,parent2,parent3,child);
						printf("%u,%u,%u>>%u\n",parent2,parent3,child,parent1);
						printf("%u,%u,%u>>%u\n",parent1,parent3,child,parent2);
						printf("%u,%u,%u>>%u\n",parent1,parent2,child,parent3);
					}

					if (!generalizationSet[child]) {

						const CatValue childVal = inst.getCatVal(order_[child]);

						for (CatValue y = 0; y < noClasses_; y++) {



							InstanceCount parentChildYCount =
									xySubDist.getCount(child, childVal, y);

							InstanceCount parentYCount =
									xySubDistParent12.getCount(order_[parent3],
											parent3Val, y);

							InstanceCount parent23YChildCount =
									xySubDistParent23.getCount(order_[child],
											childVal, y);
							InstanceCount parent13YChildCount =
									xySubDistParent13.getCount(order_[child],
											childVal, y);
							InstanceCount parent12YChildCount =
									xySubDistParent12.getCount(order_[child],
											childVal, y);
							double temp;
							if (empiricalMEst_) {

								if (xxxydist->getCount(order_[parent1],
										parent1Val, order_[parent2], parent2Val,
										order_[parent3], parent3Val) > 0) {

									temp = empiricalMEstimate(parentChildYCount,
											parentYCount,
											xydist->p(order_[child], childVal));
									spodeProbs[parent1][parent2][parent3][y] *=
											temp;
								}
								if (xxxydist->getCount(order_[parent2],
										parent2Val, order_[parent3], parent3Val,
										order_[child], childVal) > 0) {
									temp = empiricalMEstimate(parentChildYCount,
											parent23YChildCount,
											xydist->p(order_[parent1],
													parent1Val));
									spodeProbs[parent2][parent3][child][y] *=
											temp;
								}
								if (xxxydist->getCount(order_[parent1],
										parent1Val, order_[parent3], parent3Val,
										order_[child], childVal) > 0) {
									temp = empiricalMEstimate(parentChildYCount,
											parent13YChildCount,
											xydist->p(order_[parent2],
													parent2Val));
									spodeProbs[parent1][parent3][child][y] *=
											temp;
								}
								if (xxxydist->getCount(order_[parent1],
										parent1Val, order_[parent2], parent2Val,
										order_[child], childVal) > 0) {
									temp = empiricalMEstimate(parentChildYCount,
											parent12YChildCount,
											xydist->p(order_[parent3],
													parent3Val));
									spodeProbs[parent1][parent2][child][y] *=
											temp;
								}

							} else {

								if (xxxydist->getCount(order_[parent1],
										parent1Val, order_[parent2], parent2Val,
										order_[parent3], parent3Val) > 0) {
									temp = mEstimate(parentChildYCount,
											parentYCount,
											xxxxyDist_.getNoValues(child));

									spodeProbs[parent1][parent2][parent3][y] *=
											temp;

								}
								if (xxxydist->getCount(order_[parent2],
										parent2Val, order_[parent3], parent3Val,
										order_[child], childVal) > 0) {
									temp = mEstimate(parentChildYCount,
											parent23YChildCount,
											xxxxyDist_.getNoValues(parent1));
									spodeProbs[parent2][parent3][child][y] *=
											temp;
								}
								if (xxxydist->getCount(order_[parent1],
										parent1Val, order_[parent3], parent3Val,
										order_[child], childVal) > 0) {
									temp = mEstimate(parentChildYCount,
											parent13YChildCount,
											xxxxyDist_.getNoValues(parent2));
									spodeProbs[parent1][parent3][child][y] *=
											temp;
								}
								if (xxxydist->getCount(order_[parent1],
										parent1Val, order_[parent2], parent2Val,
										order_[child], childVal) > 0) {
									temp = mEstimate(parentChildYCount,
											parent12YChildCount,
											xxxxyDist_.getNoValues(parent3));
									spodeProbs[parent1][parent2][child][y] *=
											temp;
								}

							}

						}

					}
				}
			}
		}
	}



	//deal with the unselected attributes as the child
	for (CatValue parent1 = 2; parent1 < noSelectedCatAtts_; parent1++) {
		if (generalizationSet[parent1])
			continue;
		const CatValue parent1Val = inst.getCatVal(order_[parent1]);

		for (CatValue parent2 = 1; parent2 < parent1; parent2++) {
			if (generalizationSet[parent2])
				continue;
			const CatValue parent2Val = inst.getCatVal(order_[parent2]);

			XXYSubDist xxySubDistRest(
					xxxxyDist_.getXXYSubDistRest(parent1, parent1Val, parent2,
							parent2Val), noClasses_);

			XYSubDist xySubDistParent12(
					xxxydist->getXYSubDist(order_[parent1], parent1Val,
							order_[parent2], parent2Val), noClasses_);

			for (CatValue parent3 = 0; parent3 < parent2; parent3++) {
				if (generalizationSet[parent3])
					continue;
				const CatValue parent3Val = inst.getCatVal(order_[parent3]);

				XYSubDist xySubDistRest(
						xxySubDistRest.getXYSubDist(parent3, parent3Val,
								noUnSelectedCatAtts_), noClasses_);

				//check if the parent is qualified
				if (xxxydist->getCount(order_[parent1],
						parent1Val, order_[parent2], parent2Val,
						order_[parent3], parent3Val) == 0)
					continue;

				for (CatValue child = 0; child < noUnSelectedCatAtts_; child++) {

					if(verbosity>=2&&count_==1)
					{
						printf("%u,%u,%u>>%u\n",parent1,parent2,parent3,child + noSelectedCatAtts_);
					}

					if (!generalizationSet[child]) {

						const CatValue childVal = inst.getCatVal(
								order_[child + noSelectedCatAtts_]);

						for (CatValue y = 0; y < noClasses_; y++) {

							InstanceCount parentChildYCount =
									xySubDistRest.getCount(child, childVal, y);

							InstanceCount parentYCount =
									xySubDistParent12.getCount(order_[parent3],
											parent3Val, y);
							double temp;
							if (empiricalMEst_) {

								temp = empiricalMEstimate(parentChildYCount,
										parentYCount,
										xydist->p(order_[child], childVal));
								spodeProbs[parent1][parent2][parent3][y] *=
										temp;

							} else {

								temp = mEstimate(parentChildYCount, parentYCount,
										xxxxyDist_.getNoValues(
												child + noSelectedCatAtts_));
								spodeProbs[parent1][parent2][parent3][y] *=
										temp;
							}
						}
					}
				}
			}
		}
	}

	for (CatValue parent1 = 2; parent1 < noSelectedCatAtts_; parent1++) {
		for (CatValue parent2 = 1; parent2 < parent1; parent2++) {
			for (CatValue parent3 = 0; parent3 < parent2; parent3++) {
				for (CatValue y = 0; y < noClasses_; y++) {
					classDist[y] += spodeProbs[parent1][parent2][parent3][y];
				}
			}
		}
	}

	normalise(classDist);

}

void a3de3::a2deClassify(const instance &inst, std::vector<double> &classDist,
		xxxyDist & xxxyDist_) {
	CatValue delta = 0;

//	 scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max()
			/ ((noCatAtts_ - 1) * (noCatAtts_ - 2) / 2.0);

	for (CatValue father = 0; father < noCatAtts_; father++) {

		//select the attribute according to mutual information as father
		if (active_[father] == false)
			continue;

		if (!generalizationSet[father]) {

			for (CatValue mother = 0; mother < father; mother++) {

				//select the attribute according to mutual information as mother
				if (active_[mother] == false)
					continue;

				if (!generalizationSet[mother]) {
					CatValue parent = 0;
					for (CatValue y = 0; y < noClasses_; y++) {
						parent += xxxyDist_.xxyCounts.getCount(father,
								inst.getCatVal(father), mother,
								inst.getCatVal(mother), y);
					}
					if (parent > 0) {

						delta++;

						for (CatValue y = 0; y < noClasses_; y++) {
							double p = weight_a2de[father][mother]
									* xxxyDist_.xxyCounts.jointP(father,
											inst.getCatVal(father), mother,
											inst.getCatVal(mother), y)
									* scaleFactor; // scale up by maximum possible factor to reduce risk of numeric underflow

							for (CatValue child = 0; child < noCatAtts_;
									child++) {

								if (!generalizationSet[child]) {
									if (child != father && child != mother)
										p *= xxxyDist_.unorderedP(child,
												inst.getCatVal(child), father,
												inst.getCatVal(father), mother,
												inst.getCatVal(mother), y);
								}
							}
							classDist[y] += p;
							if (verbosity >= 3) {
								printf("%f,", classDist[y]);
							}
						}
						if (verbosity >= 3) {
							printf("<<<<\n");
						}
					}
				}
			}
		}
		if (verbosity >= 3) {
			printf(">>>>>>>\n");
		}
	}
	if (delta == 0) {
		aodeClassify(inst, classDist, xxxyDist_.xxyCounts);
	} else

		normalise(classDist);
	if (verbosity >= 3) {
		print(classDist);
		printf("\n");
	}

}

void a3de3::aodeClassify(const instance &inst, std::vector<double> &classDist,
		xxyDist & xxyDist_) {

	// scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max() / noCatAtts_;

	CatValue delta = 0;

	for (CategoricalAttribute parent = 0; parent < noCatAtts_; parent++) {

		//select the attribute according to mutual information as parent
		if (active_[parent] == false)
			continue;

		if (!generalizationSet[parent]) {

			if (xxyDist_.xyCounts.getCount(parent, inst.getCatVal(parent))
					> 0) {

				delta++;
				for (CatValue y = 0; y < noClasses_; y++) {
					double p = weight_aode[parent]
							* xxyDist_.xyCounts.jointP(parent,
									inst.getCatVal(parent), y) * scaleFactor;

					for (CategoricalAttribute child = 0; child < noCatAtts_;
							child++) {

						//should select child using the conditional mutual information on parent
						//						if (active_[child] == false)
						//							continue;

						if (!generalizationSet[child]) {
							if (child != parent)
								p *= xxyDist_.p(child, inst.getCatVal(child),
										parent, inst.getCatVal(parent), y);
						}
					}
					classDist[y] += p;
				}
			}
		}
	}

	if (delta == 0) {
		nbClassify(inst, classDist, xxyDist_.xyCounts);
	} else
		normalise(classDist);

}
void a3de3::nbClassify(const instance &inst, std::vector<double> &classDist,
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

