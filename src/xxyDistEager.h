/* Open source system for classification learning from very large data
 ** Copyright (C) 2012 Geoffrey I Webb
 ** Class for handling a joint distribution between two attributes and a class
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

#include "instanceStream.h"
#include "xyDist.h"
#include "xxyDist.h"

class xxyDistEager{
public:
	xxyDistEager();
	xxyDistEager(InstanceStream& stream);
	~xxyDistEager(void);

	void calculateCondProb();

	void reset(InstanceStream& stream);

	void update(const instance& i);

	// p(x1=v1, x2=v2, Y=y) unsmoothed
	inline double rawJointP(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2, CatValue y) const {
		return (*constRef(x1, v1, x2, v2, y)) / (xyCounts.count);
	}

	// p(x1=v1, x2=v2, Y=y) using M-estimate
	inline double jointP(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2, CatValue y) const {
		return (*constRef(x1, v1, x2, v2, y)
				+ M
						/ (instanceStream_->getNoValues(x1)
								* instanceStream_->getNoValues(x2)
								* instanceStream_->getNoClasses()))
				/ (xyCounts.count + M);
	}

	// p(x1=v1, x2=v2) using M-estimate
	inline double jointP(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2) const {
		return (getCount(x1, v1, x2, v2)
				+ M
						/ (instanceStream_->getNoValues(x1)
								* instanceStream_->getNoValues(x2)))
				/ (xyCounts.count + M);
	}

	// p(x1=v1|Y=y, x2=v2) using M-estimate
	inline double p(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2, CatValue y) const {
		return (*constRef(x1, v1, x2, v2, y)
				+ M / instanceStream_->getNoValues(x1))
				/ (xyCounts.getCount(x2, v2, y) + M);
	}

	//inline double p(CatValue y) const {
	//  return xyCounts.p(y);
	//}

	// p(x1=v1, x2=v2, Y=y) unsmoothed
	inline InstanceCount getCount(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2, CatValue y) const {
		return *constRef(x1, v1, x2, v2, y);
	}

	// count for instances x1=v1, x2=v2
	inline InstanceCount getCount(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2) const {
		InstanceCount c = 0;

		const unsigned int noClasses = instanceStream_->getNoClasses();

		for (CatValue y = 0; y < noClasses; y++) {
			c += getCount(x1, v1, x2, v2, y);
		}
		return c;
	}

private:
	// count[X1=x1][X2=x2][Y=y]
	inline InstanceCount *ref(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2, CatValue y) {
		if (x2 > x1) {
			CatValue t = x1;
			x1 = x2;
			x2 = t;
			t = v1;
			v1 = v2;
			v2 = t;
		}

		return &count[x1][v1][x2][v2 ][ y];

	}

//  // count[X1=x1][X2=x2]
//  inline InstanceCount *xxref(CategoricalAttribute x1, CatValue v1, CategoricalAttribute x2, CatValue v2) {
//    if (x2 > x1) {
//      CatValue t = x1;
//      x1 = x2;
//      x2 = t;
//      t = v1;
//      v1 = v2;
//      v2 = t;
//    }
//
//    return &count[offset2[x1]+v1*offset1[x1]+offset1[x2]+v2*instanceStream_->getNoClasses()];
//  }

	// count[X1=x1][X2=x2][Y=y]
	inline const InstanceCount *constRef(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2, CatValue y) const {
		if (x2 > x1) {
			CatValue t = x1;
			x1 = x2;
			x2 = t;
			t = v1;
			v1 = v2;
			v2 = t;
		}

		return &count[x1][v1][x2][v2 ][ y];
	}

public:
	InstanceStream* instanceStream_;
	//std::vector<InstanceCount> count;    // a one-dimensional flattened representation of count[X1=x1][X2=x2][Y=y] storing only X1 < X2
	std::vector<std::vector<std::vector<std::vector<std::vector<InstanceCount> > > > >  count;

	std::vector<std::vector<std::vector<std::vector<std::vector<double> > > > > condiProbs;

	unsigned int noCatAtts_; ///< the number of categorical CategoricalAttributes.

	unsigned int noClasses_;  ///< the number of classes
	xyDist xyCounts;

private:
	// std::vector<unsigned int> offset1;  // The offset for an Attribute in one-dimensional space
	// std::vector<unsigned int> offset2;  // The offset for an Attribute in two-dimensional space
	//unsigned int countSize;
};
