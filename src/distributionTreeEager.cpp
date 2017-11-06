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
#include "distributionTreeEager.h"
#include "smoothing.h"
#include "utils.h"
#include <assert.h>

InstanceStream const *dtNodeEager::instanceStream_;

dtNodeEager::dtNodeEager(InstanceStream const* stream, const CategoricalAttribute a) :
		att(NOPARENTEAGER), xyCount(stream->getNoValues(a),
				stream->getNoClasses()), condProbs(
				instanceStream_->getNoValues(a),
				instanceStream_->getNoClasses()) {
	instanceStream_ = stream;
}

dtNodeEager::dtNodeEager(const CategoricalAttribute a) :
		att(NOPARENTEAGER), xyCount(instanceStream_->getNoValues(a),
				instanceStream_->getNoClasses()), condProbs(
				instanceStream_->getNoValues(a),
				instanceStream_->getNoClasses()) {
}

dtNodeEager::dtNodeEager() :
		att(NOPARENTEAGER) {
}

dtNodeEager::~dtNodeEager() {
}

void dtNodeEager::init(InstanceStream const* stream,
		const CategoricalAttribute a) {
	instanceStream_ = stream;
	att = NOPARENTEAGER;
	xyCount.assign(instanceStream_->getNoValues(a),
			instanceStream_->getNoClasses(), 0);
	condProbs.assign(instanceStream_->getNoValues(a),
			instanceStream_->getNoClasses(), 0);
	children.clear();
}

void dtNodeEager::clear(CategoricalAttribute a) {
	xyCount.clear();
	condProbs.clear();
	children.clear();
	att = NOPARENTEAGER;
}

distributionTreeEager::distributionTreeEager() {
}

distributionTreeEager::distributionTreeEager(InstanceStream const* stream,
		const CategoricalAttribute att) :
		instanceStream_(stream), dTree(stream, att) {
}

distributionTreeEager::~distributionTreeEager(void) {
}

void distributionTreeEager::init(InstanceStream const& stream,
		const CategoricalAttribute att) {
	instanceStream_ = &stream;
	dTree.init(&stream, att);
}

void distributionTreeEager::clear(CategoricalAttribute a) {
	dTree.clear(a);
}

void distributionTreeEager::findLeaf(dtNodeEager *dt,
		const CategoricalAttribute att) {

	if (dt == NULL)
		return;
	else if (dt->children.empty()) {

		// sum over all values of the Attribute for the class to obtain count[y, parents]
		for (CatValue value = 0; value < instanceStream_->getNoValues(att);
				value++) {

			for (CatValue y = 0; y < instanceStream_->getNoClasses(); y++) {

				InstanceCount totalCount = dt->getCount(0, y);
				for (CatValue v = 1; v < instanceStream_->getNoValues(att);
						v++) {
					totalCount += dt->getCount(v, y);
				}

				//		classDist[y] *= 1.0;
				dt->condProbs.ref(value, y) = mEstimate(dt->getCount(value, y),
						totalCount, instanceStream_->getNoValues(att));
//				dt->condProbs.ref(value, y) =1.0;
			}

		}

	} else {
		for (CatValue v = 0; v < dt->children.size(); v++) {
			findLeaf(dt->children[v], att);
		}
	}
}
void distributionTreeEager::calculateProbs(const CategoricalAttribute att) {
	findLeaf(&dTree, att);
}

void distributionTreeEager::update(const instance &i, const CategoricalAttribute a,
		const std::vector<CategoricalAttribute> &parents,
		const unsigned int k) {
	const CatValue y = i.getClass();
	const CatValue v = i.getCatVal(a);

	dTree.ref(v, y)++;

	dtNodeEager
	*currentNode = &dTree;

	for (unsigned int d = 0; d < parents.size(); d++) {

		const CategoricalAttribute p = parents[d];

		if (currentNode->att == NOPARENTEAGER || currentNode->children.empty()) {
			// children array has not yet been allocated
			currentNode->children.assign(instanceStream_->getNoValues(p), NULL);
			currentNode->att = p;
		}

		assert(currentNode->att == p);

		dtNodeEager *nextNode = currentNode->children[i.getCatVal(p)];

		// the child has not yet been allocated, so allocate it
		if (nextNode == NULL) {
			currentNode = currentNode->children[i.getCatVal(p)] = new dtNodeEager(
					a);
		} else {
			currentNode = nextNode;
		}

		currentNode->ref(v, y)++;}
	}

// update classDist using the evidence from the tree about i
void distributionTreeEager::updateClassDistribution(std::vector<double> &classDist,
		const CategoricalAttribute a, const instance &i) {
	dtNodeEager *dt = &dTree;
	CategoricalAttribute att = dTree.att;

	// find the appropriate leaf
	while (att != NOPARENTEAGER) {
		const CatValue v = i.getCatVal(att);
		dtNodeEager *next = dt->children[v];
		if (next == NULL)
			break;
		dt = next;
		att = dt->att;
	}

	// sum over all values of the Attribute for the class to obtain count[y, parents]
	for (CatValue y = 0; y < instanceStream_->getNoClasses(); y++) {
//		InstanceCount totalCount = dt->getCount(0, y);
//		for (CatValue v = 1; v < instanceStream_->getNoValues(a); v++) {
//			totalCount += dt->getCount(v, y);
//		}
//
////		classDist[y] *= 1.0;
//		classDist[y] *= mEstimate(dt->getCount(i.getCatVal(a), y), totalCount,
//				instanceStream_->getNoValues(a));
		classDist[y] *= dt->condProbs.ref(i.getCatVal(a), y);

	}
}

// update classDist using the evidence from the tree about i
// require that at least minCount values be used for proability estimation
void distributionTreeEager::updateClassDistribution(std::vector<double> &classDist,
		const CategoricalAttribute a, const instance &i,
		InstanceCount minCount) {
	dtNodeEager *dt = &dTree;
	CategoricalAttribute att = dTree.att;

	// find the appropriate leaf
	while (att != NOPARENTEAGER) {
		const CatValue v = i.getCatVal(att);
		dtNodeEager *next = dt->children[v];
		if (next == NULL)
			break;

		// check that the next node has enough examples for this value;
		InstanceCount cnt = 0;
		for (CatValue y = 0; y < instanceStream_->getNoClasses(); y++) {
			cnt += next->getCount(i.getCatVal(a), y);
		}

		if (cnt < minCount)
			break;

		dt = next;
		att = dt->att;
	}

	// sum over all values of the Attribute for the class to obtain count[y, parents]
	for (CatValue y = 0; y < instanceStream_->getNoClasses(); y++) {
		InstanceCount totalCount = dt->getCount(0, y);
		for (CatValue v = 1; v < instanceStream_->getNoValues(a); v++) {
			totalCount += dt->getCount(v, y);
		}

		classDist[y] *= mEstimate(dt->getCount(i.getCatVal(a), y), totalCount,
				instanceStream_->getNoValues(a));
	}
}

// update classDist using the evidence from the tree about i and deducting it at the same time (Pazzani's trick for loocv)
// require that at least 1 value (minCount = 1) be used for probability estimation
void distributionTreeEager::updateClassDistributionloocv(
		std::vector<double> &classDist, const CategoricalAttribute a,
		const instance &i) {
	dtNodeEager *dt = &dTree;
	CategoricalAttribute att = dTree.att;

	// find the appropriate leaf
	while (att != NOPARENTEAGER) {
		const CatValue v = i.getCatVal(att);
		dtNodeEager *next = dt->children[v];
		if (next == NULL)
			break;

		// check that the next node has enough examples for this value;
		InstanceCount cnt = 0;
		for (CatValue y = 0; y < instanceStream_->getNoClasses(); y++) {
			cnt += next->getCount(i.getCatVal(a), y);
		}

		//In loocv, we consider minCount=1(+1), since we have to leave out i.
		if (cnt < 2)
			break;

		dt = next;
		att = dt->att;
	}

	// sum over all values of the Attribute for the class to obtain count[y, parents]
	for (CatValue y = 0; y < instanceStream_->getNoClasses(); y++) {
		InstanceCount totalCount = dt->getCount(0, y);
		for (CatValue v = 1; v < instanceStream_->getNoValues(a); v++) {
			totalCount += dt->getCount(v, y);
		}

		if (y != i.getClass())
			classDist[y] *= mEstimate(dt->getCount(i.getCatVal(a), y),
					totalCount, instanceStream_->getNoValues(a));
		else
			classDist[y] *= mEstimate(dt->getCount(i.getCatVal(a), y) - 1,
					totalCount - 1, instanceStream_->getNoValues(a));
	}
}

void dtNodeEager::updateStats(CategoricalAttribute target,
		std::vector<CategoricalAttribute> &parents, unsigned int k,
		unsigned int depth, unsigned long long int &pc, double &apd,
		unsigned long long int &zc) {
	if (depth == parents.size() || children.empty()) {
		for (CatValue v = 0; v < instanceStream_->getNoValues(target); v++) {
			pc++;

			apd += (depth - apd) / static_cast<double>(pc);

			for (CatValue y = 0; y < instanceStream_->getNoClasses(); y++) {
				if (getCount(v, y) == 0)
					zc++;
			}
		}
	} else {
		for (CatValue v = 0; v < instanceStream_->getNoValues(parents[depth]);
				v++) {
			if (children[v] == NULL) {
				unsigned long int pathsMissing = 1;

				for (unsigned int i = depth; i < parents.size(); i++)
					pathsMissing *= instanceStream_->getNoValues(parents[i]);

				pc += pathsMissing;

				apd += pathsMissing
						* ((depth - apd) / (pc - pathsMissing / 2.0));

				for (CatValue tv = 0; tv < instanceStream_->getNoValues(target);
						tv++) {
					for (CatValue y = 0; y < instanceStream_->getNoClasses();
							y++) {
						if (getCount(tv, y) == 0)
							zc++;
					}
				}
			} else {
				children[v]->updateStats(target, parents, k, depth + 1, pc, apd,
						zc);
			}
		}
	}
}

void distributionTreeEager::updateStats(CategoricalAttribute target,
		std::vector<CategoricalAttribute> &parents, unsigned int k,
		unsigned long long int &pc, double &apd, unsigned long long int &zc) {
	//dTree.updateStats(target, parents, parents.size(), 1, pc, apd, zc);
}
