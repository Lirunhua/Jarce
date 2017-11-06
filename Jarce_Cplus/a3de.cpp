/*
 * a3de.cpp
 *
 *  Created on: 28/09/2012
 *      Author: shengleichen
 */

#include "a3de.h"
#include "assert.h"
#include "utils.h"
#include <algorithm>
#include "correlationMeasures.h"
#include "globals.h"
#include "instanceStream.h"

a3de::a3de(char* const *& argv, char* const * end) :
		weight_a2de(1) {
	name_ = "a3de";

	// TODO Auto-generated constructor stub
	weighted = false;
	minCount = 100;
	subsumptionResolution = false;
	selected = false;

	selected = false;
	oneSelective_ = false;
	twoSelective_ = false;

	chisq_ = false;
	acmi_ = false;
	mi_ = false;
	su_ = false;
	sum_ = false;
	avg_ = false;
	factor_ = 1;  //default value

	memorySelective_ = false;

	// get arguments
	while (argv != end) {
		if (*argv[0] != '-') {
			break;
		} else if (streq(argv[0] + 1, "w")) {
			weighted = true;
		} else if (streq(argv[0] + 1, "sub")) {
			subsumptionResolution = true;

		} else if (streq(argv[0] + 1, "selective")) {
			selected = true;

		} else if (streq(argv[0] + 1, "chisq")) {
			selected = true;
			chisq_ = true;
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
		} else if (argv[0][1] == 'c') {
			getUIntFromStr(argv[0] + 2, minCount, "c");
		} else {
			error("a3de does not support argument %s\n", argv[0]);
			break;
		}

		name_ += *argv;

		++argv;
	}

	if (selected == true) {
		if (mi_ == false && su_ == false && chisq_ == false)
			chisq_ = true;
	}

	trainingIsFinished_ = false;
}

a3de::~a3de() {
	// TODO Auto-generated destructor stub
}

void a3de::reset(InstanceStream &is) {
	xxxxyDist_.reset(is);
	trainingIsFinished_ = false;

	noCatAtts_ = is.getNoCatAtts();
	noClasses_ = is.getNoClasses();
	inactiveCnt_ = 0;

	weight_aode.assign(noCatAtts_, 1);
	weight_a2de = crosstab<double>(noCatAtts_);
	weight = crosstab3D<double>(noCatAtts_);


	//initialise the weight for non-weighting

	weight_aode.assign(noCatAtts_, 1);
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
//
//	generalizationSet.assign(noCatAtts_, -1);
//	substitutionSet.assign(noCatAtts_, -1);

	generalizationSet.assign(noCatAtts_, false);

	active_.assign(noCatAtts_, true);
	instanceStream_ = &is;

}

void a3de::getCapabilities(capabilities &c) {
	c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void a3de::initialisePass() {

}

/// true iff no more passes are required. updated by finalisePass()
bool a3de::trainingIsFinished() {
	return trainingIsFinished_;
}

void a3de::train(const instance &inst) {
	xxxxyDist_.update(inst);
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

void a3de::finalisePass() {

	trainingIsFinished_ = true;


	if (weighted) {

		// for aode
		weight_aode.assign(noCatAtts_, 0);
		for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {

			for (CatValue x = 0; x < xxxxyDist_.instanceStream_->getNoValues(i);
					x++) {
				for (CatValue y = 0; y < noClasses_; y++) {
					double pXy =
							xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.jointP(i,
									x, y);
					if (pXy == 0)
						continue;

					double pY = xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.p(y);

					double pX = xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.p(i,
							x);

					double weightXy = pXy * log2(pXy / (pX * pY));
					weight_aode[i] += weightXy;
				}
			}
		}

		//for a2de
		//use mutualExt to weight
		xxyDist dist = xxxxyDist_.xxxyCounts.xxyCounts;
		for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
			for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

				double m = 0.0;
				double n = 0.0;

				for (CatValue v1 = 0; v1 < instanceStream_->getNoValues(x1);
						v1++) {
					for (CatValue v2 = 0; v2 < instanceStream_->getNoValues(x2);
							v2++) {
						for (CatValue y = 0; y < noClasses_; y++) {

							const double px1x2y =
									xxxxyDist_.xxxyCounts.xxyCounts.jointP(x1,
											v1, x2, v2, y);

							if (verbosity >= 4) {
								printf("%d\t%" ICFMT "\n\t%" ICFMT "\n\t%f\n",
										y,
										xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.getClassCount(
												y), dist.xyCounts.count,
										xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.p(
												y));
								printf("%d,%d,%d,%f\n", v1, v2, y, px1x2y);
							}

							if (px1x2y) {
								n =
										px1x2y
												* log2(
														px1x2y
																/ (xxxxyDist_.xxxyCounts.xxyCounts.jointP(
																		x1, v1,
																		x2, v2)
																		* xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.p(
																				y)));
								m += n;
								if (verbosity >= 4)
									if (x1 == 2 && x2 == 0) {
										printf("%e\t%e\t%f\n", px1x2y,
												xxxxyDist_.xxxyCounts.xxyCounts.jointP(
														x1, v1, x2, v2),
												xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.p(
														y));
										printf("%e\n", n);
									}
							}
						}
					}
				}
				assert(m >= -0.00000001);
				// CMI is always positive, but allow for some imprecision
				weight_a2de[x1][x2] = m;
				weight_a2de[x2][x1] = m;
				//printf("%d,%d\t%f\n", x1, x2, m);
			}
		}

		//for a3de
		for (CategoricalAttribute x1 = 2; x1 < noCatAtts_; x1++) {
			for (CategoricalAttribute x2 = 1; x2 < x1; x2++) {
				for (CategoricalAttribute x3 = 0; x3 < x2; x3++) {
					double m = 0.0;
					double n = 0.0;

					for (CatValue v1 = 0; v1 < instanceStream_->getNoValues(x1);
							v1++) {
						for (CatValue v2 = 0;
								v2 < instanceStream_->getNoValues(x2); v2++) {
							for (CatValue v3 = 0;
									v3 < instanceStream_->getNoValues(x3);
									v3++) {

								for (CatValue y = 0; y < noClasses_; y++) {
									const double px1x2x3y =
											xxxxyDist_.xxxyCounts.jointP(x1, v1,
													x2, v2, x3, v3, y);
									if (px1x2x3y == 0)
										continue;
									n =
											px1x2x3y
													* log2(
															px1x2x3y
																	/ (xxxxyDist_.xxxyCounts.jointP(
																			x1,
																			v1,
																			x2,
																			v2,
																			x3,
																			v3)
																			* xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.p(
																					y)));
									m += n;
								}

							}
						}

					}

					assert(m >= -0.00000001);
					// CMI is always positive, but allow for some imprecision
					weight[x1][x2][x3] = m;
					weight[x2][x1][x3] = m;
					weight[x1][x3][x2] = m;
					weight[x2][x3][x1] = m;
					weight[x3][x1][x2] = m;
					weight[x3][x2][x1] = m;
				}
			}
		}
	}

	if (selected) {

		// sort the attributes on symmetrical uncertainty with the class

		std::vector<CategoricalAttribute> order_;

		for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
			order_.push_back(a);
		}
		xxyDist &xxyDist_=xxxxyDist_.xxxyCounts.xxyCounts;

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

			unsigned int noSelectedCatAtts_;
			unsigned int v;
			unsigned int i;
			v = instanceStream_->getNoValues(0);
			for (i = 1; i < noCatAtts_; i++)
				v += instanceStream_->getNoValues(i);
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

		}else if  (chisq_ == true) {

			bool flag = true;
			double lowest;
			CategoricalAttribute attLowest;

			for (std::vector<CategoricalAttribute>::const_iterator it =
					order_.begin(); it != order_.end(); it++) {

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
							tab[r * cols + c] +=
									xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.getCount(a, r,
											c);
						}
					}

					double critVal = 0.05 / noCatAtts_;
					double chisqVal = chiSquare(tab, rows, cols);

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

void a3de::classify(const instance &inst, std::vector<double> &classDist) {

	generalizationSet.assign(noCatAtts_, false);

	//compute the generalisation set and substitution set for
	//lazy subsumption resolution
	if (subsumptionResolution == true) {
		for (CategoricalAttribute i = 1; i < noCatAtts_; i++) {
			for (CategoricalAttribute j = 0; j < i; j++) {
				if (!generalizationSet[j]) {
					InstanceCount countOfxixj =
							xxxxyDist_.xxxyCounts.xxyCounts.getCount(i,
									inst.getCatVal(i), j, inst.getCatVal(j));
					InstanceCount countOfxj =
							xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.getCount(j,
									inst.getCatVal(j));
					InstanceCount countOfxi =
							xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.getCount(i,
									inst.getCatVal(i));

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

	for (CatValue y = 0; y < noClasses_; y++)
		classDist[y] = 0;

	CatValue delta = 0;

	// scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max()
			/ ((noCatAtts_ - 1) * (noCatAtts_ - 2) * (noCatAtts_) / 6.0);

	for (CatValue parent1 = 2; parent1 < noCatAtts_; parent1++) {

		//select the attribute according to mutual information as father
		if (active_[parent1] == false)
			continue;

		if (!generalizationSet[parent1]) {

			for (CatValue parent2 = 1; parent2 < parent1; parent2++) {

				//select the attribute according to mutual information as parent2
				if (active_[parent2] == false)
					continue;

				if (!generalizationSet[parent2]) {

					for (CatValue parent3 = 0; parent3 < parent2; parent3++) {

						//select the attribute according to mutual information as father
						if (active_[parent3] == false)
							continue;

						if (!generalizationSet[parent3]) {

							CatValue parent = 0;
							for (CatValue y = 0; y < noClasses_; y++) {
								parent += xxxxyDist_.xxxyCounts.getCount(
										parent1, inst.getCatVal(parent1),
										parent2, inst.getCatVal(parent2),
										parent3, inst.getCatVal(parent3), y);
							}
							if (parent > 0) {

								delta++;

								for (CatValue y = 0; y < noClasses_; y++) {

									//weight[parent1][parent2]*
									double p = weight[parent1][parent2][parent3]
											* xxxxyDist_.xxxyCounts.jointP(
													parent1,
													inst.getCatVal(parent1),
													parent2,
													inst.getCatVal(parent2),
													parent3,
													inst.getCatVal(parent3), y)
											* scaleFactor;// scale up by maximum possible factor to reduce risk of numeric underflow

									for (CatValue child = 0; child < noCatAtts_;
											child++) {

										if (!generalizationSet[child]) {
											if (child != parent1
													&& child != parent2
													&& child != parent3)
												p *= xxxxyDist_.p(child,
														inst.getCatVal(child),
														parent1,
														inst.getCatVal(parent1),
														parent2,
														inst.getCatVal(parent2),
														parent3,
														inst.getCatVal(parent3),
														y);
										}
									}
									classDist[y] += p;
								}
							}
						}
					}
				}
			}

		}
	}
	if (delta == 0) {

		a2deClassify(inst, classDist, xxxxyDist_.xxxyCounts);
	} else

		normalise(classDist);
}

void a3de::a2deClassify(const instance &inst, std::vector<double> &classDist,
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

void a3de::aodeClassify(const instance &inst, std::vector<double> &classDist,
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
void a3de::nbClassify(const instance &inst, std::vector<double> &classDist,
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

