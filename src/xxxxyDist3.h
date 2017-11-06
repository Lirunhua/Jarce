#pragma once

#include "xxxyDist.h"
#include "assert.h"

class xxxxyDist3 {
public:
	xxxxyDist3();
	xxxxyDist3(InstanceStream& stream);
	~xxxxyDist3(void);

	void reset(InstanceStream& stream);

	void update(const instance& i);

	void clear();
//
//	// p(x1=v1, x2=v2, x3=v3, x4=v4, Y=y) unsmoothed
//	inline double rawJointP(CategoricalAttribute x1, CatValue v1,
//			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
//			CatValue v3, CategoricalAttribute x4, CatValue v4,
//			CatValue y) const {
//		return (*constRef(x1, v1, x2, v2, x3, v3, x4, v4, y))
//				/ (xxxyCounts.xxyCounts.xyCounts.count);
//	}
//
//	// p(x1=v1, x2=v2, x3=v3,x4=v4, Y=y) using M-estimate
//	inline double jointP(CategoricalAttribute x1, CatValue v1,
//			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
//			CatValue v3, CategoricalAttribute x4, CatValue v4,
//			CatValue y) const {
//		return (*constRef(x1, v1, x2, v2, x3, v3, x4, v4, y)
//				+ M
//						/ (instanceStream_->getNoValues(x1)
//								* instanceStream_->getNoValues(x2)
//								* instanceStream_->getNoValues(x3) * noClasses_))
//				/ (xxxyCounts.xxyCounts.xyCounts.count + M);
//	}
//
//// p(x1=v1|Y=y, x2=v2, x3=v3,x4=v4) using M-estimate
//	inline double p(CategoricalAttribute x1, CatValue v1,
//			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
//			CatValue v3, CategoricalAttribute x4, CatValue v4,
//			CatValue y) const {
//
//		return (*constRef(x1, v1, x2, v2, x3, v3, x4, v4, y)
//				+ M / instanceStream_->getNoValues(x1))
//				/ (xxxyCounts.getCount(x2, v2, x3, v3, x4, v4, y) + M);
//	}
//
//	// get count for instance (x1=v1, x2=v2, x3=v3,x4=v4, Y=y)
//	inline InstanceCount getCount(CategoricalAttribute x1, CatValue v1,
//			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
//			CatValue v3, CategoricalAttribute x4, CatValue v4,
//			CatValue y) const {
//		return *constRef(x1, v1, x2, v2, x3, v3, x4, v4, y);
//	}

	inline std::vector<std::vector<InstanceCount> >* getXXYSubDist(
			CategoricalAttribute x1, CatValue v1, CategoricalAttribute x2,
			CatValue v2) {
		return &countSelected[x1][v1 * x1 + x2][v2 * x2];
	}

	inline std::vector<std::vector<InstanceCount> >* getXXYSubDistRest(
			CategoricalAttribute x1, CatValue v1, CategoricalAttribute x2,
			CatValue v2) {
		return &countUnSelected[x1][v1 * x1 + x2][v2 * x2];
	}

	void setOrder(std::vector<CategoricalAttribute> &order);
	void setNoSelectedCatAtts(unsigned int noSelectedCatAtts);

	inline unsigned int getNoValues(const CategoricalAttribute a) const {
		return metaData_->getNoValues(order_[a]);
	}

private:
	// count[X1=x1][X2=x2][X3=x3][X4=x4][Y=y]
	inline InstanceCount *ref(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
			CatValue v3, CategoricalAttribute x4, CatValue v4, CatValue y) {
//		if (x2 > x1) {
//			CatValue t = x1;
//			x1 = x2;
//			x2 = t;
//			t = v1;
//			v1 = v2;
//			v2 = t;
//		}
//		if (x3 > x2) {
//			CatValue t = x2;
//			x2 = x3;
//			x3 = t;
//			t = v2;
//			v2 = v3;
//			v3 = t;
//		}
//		if (x4 > x3) {
//			CatValue t = x3;
//			x3 = x4;
//			x4 = t;
//			t = v3;
//			v3 = v4;
//			v4 = t;
//		}
//
//		if (x2 > x1) {
//			CatValue t = x1;
//			x1 = x2;
//			x2 = t;
//			t = v1;
//			v1 = v2;
//			v2 = t;
//		}
//		if (x3 > x2) {
//			CatValue t = x2;
//			x2 = x3;
//			x3 = t;
//			t = v2;
//			v2 = v3;
//			v3 = t;
//		}
//		if (x2 > x1) {
//			CatValue t = x1;
//			x1 = x2;
//			x2 = t;
//			t = v1;
//			v1 = v2;
//			v2 = t;
//		}

		assert(x1 > x2 && x2 >x3&& x3>x4);
		return &countSelected[x1][v1 * x1 + x2][v2 * x2 + x3][v3 * x3 + x4][v4
				* noClasses_ + y];
	}
	inline InstanceCount *refRest(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
			CatValue v3, CategoricalAttribute x4, CatValue v4, CatValue y) {

		assert(x1 > x2 && x2 >x3);

		return &countUnSelected[x1][v1 * x1 + x2][v2 * x2 + x3][v3
				* noUnSelectedCatAtts_ + x4][v4 * noClasses_ + y];
	}
//	// count[X1=x1][X2=x2][X3=x3]
//	inline std::vector<instanceCount> *xxxref(CategoricalAttribute x1, CatValue v1,
//			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3, CatValue v3) {
//		assert(x1 < x2 && x2 < x3);
//
//		return &count[x1][v1][x2][v2][x3][v3];
//	}

// count[X1=x1][X2=x2][X3=x3][Y=y]
//	inline const InstanceCount *constRef(CategoricalAttribute x1, CatValue v1,
//			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
//			CatValue v3, CategoricalAttribute x4, CatValue v4,
//			CatValue y) const {
//		if (x2 > x1) {
//			CatValue t = x1;
//			x1 = x2;
//			x2 = t;
//			t = v1;
//			v1 = v2;
//			v2 = t;
//		}
//		if (x3 > x2) {
//			CatValue t = x2;
//			x2 = x3;
//			x3 = t;
//			t = v2;
//			v2 = v3;
//			v3 = t;
//		}
//		if (x4 > x3) {
//			CatValue t = x3;
//			x3 = x4;
//			x4 = t;
//			t = v3;
//			v3 = v4;
//			v4 = t;
//		}
//
//		if (x2 > x1) {
//			CatValue t = x1;
//			x1 = x2;
//			x2 = t;
//			t = v1;
//			v1 = v2;
//			v2 = t;
//		}
//		if (x3 > x2) {
//			CatValue t = x2;
//			x2 = x3;
//			x3 = t;
//			t = v2;
//			v2 = v3;
//			v3 = t;
//		}
//		if (x2 > x1) {
//			CatValue t = x1;
//			x1 = x2;
//			x2 = t;
//			t = v1;
//			v1 = v2;
//			v2 = t;
//		}
//		assert(x1 > x2 && x2 >x3&& x3>x4);
//		return &count[x1][v1 * x1 + x2][v2 * x2 + x3][v3 * x3 + x4][v4
//				* noClasses_ + y];
//	}

public:
	unsigned int noCatAtts_; ///< the number of categorical CategoricalAttributes.
	unsigned int noClasses_;  ///< the number of classes

	InstanceStream* instanceStream_;
	xxxyDist xxxyCounts;
	InstanceStream::MetaData* metaData_;

	unsigned int noSelectedCatAtts_; ///< the number of categorical CategoricalAttributes.
	unsigned int noUnSelectedCatAtts_; ///< the number of categorical CategoricalAttributes.

	std::vector<CategoricalAttribute> order_;

	// a five-dimensional semiflattened representation of count[X1=x1][X2=x2][X3=x3][Y=y] storing only X1 > X2 > X3
	// outer vector is indexed by X1
	// second vector is indexed by x1*X2
	// third vector is indexed by x2*X3
	// fourth vector is indexed by x3*X4
	// inner vector is indexed by x4*y
	std::vector<
			std::vector<std::vector<std::vector<std::vector<InstanceCount> > > > > countSelected;
	std::vector<
			std::vector<std::vector<std::vector<std::vector<InstanceCount> > > > > countUnSelected;
};
