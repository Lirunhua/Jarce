#pragma once

#include "xxyDist.h"
#include "assert.h"
#include "utils.h"
//
//class constXXYSubDist {
//public:
//  constXXYSubDist(const std::vector<std::vector<InstanceCount> >* subDist, const unsigned int noOfClasses) : subDist_(subDist), noOfClasses_(noOfClasses) {}
//
//  inline const std::vector<InstanceCount>* getXYSubDist(const CategoricalAttribute x, const CatValue v) const { return &subDist_[x][v*x]; }
//  inline const InstanceCount getCount(const CategoricalAttribute x1, const CatValue v1, const CategoricalAttribute x2, const CatValue v2, const CatValue y) const { return subDist_[x1][v1*x1+x2][v2*noOfClasses_+y]; }
//
//private:
//  const std::vector<std::vector<InstanceCount> >* subDist_;
//  const unsigned int noOfClasses_;
//};
//
//class XXYSubDist {
//public:
//  XXYSubDist(std::vector<std::vector<InstanceCount> >* subDist, const unsigned int noOfClasses) : subDist_(subDist), noOfClasses_(noOfClasses) {}
//
//  inline std::vector<InstanceCount>* getXYSubDist(const CategoricalAttribute x, const CatValue v) { return &subDist_[x][v*x]; }
//  inline InstanceCount getCount(const CategoricalAttribute x1, const CatValue v1, const CategoricalAttribute x2, const CatValue v2, const CatValue y) const { return subDist_[x1][v1*x1+x2][v2*noOfClasses_+y]; }
//  inline void incCount(const CategoricalAttribute x1, const CatValue v1, const CategoricalAttribute x2, const CatValue v2, const CatValue y) { ++subDist_[x1][v1*x1+x2][v2*noOfClasses_+y]; }
//
//  std::vector<std::vector<InstanceCount> >* subDist_;
//  const unsigned int noOfClasses_;
//};
//
//class constXXYSubDist2 {
//public:
//  constXXYSubDist2(std::vector<std::vector<InstanceCount> >* subDist, const unsigned int noOfClasses,const unsigned int noOfCatAtts) : subDist_(subDist), noOfClasses_(noOfClasses),noOfCatAtts_(noOfCatAtts) {}
//
//  inline const std::vector<InstanceCount>* getXYSubDist(const CategoricalAttribute x, const CatValue v) const { return &subDist_[x][v*noOfCatAtts_]; }
//  inline InstanceCount getCount(const CategoricalAttribute x1, const CatValue v1, const CategoricalAttribute x2, const CatValue v2, const CatValue y) const { return subDist_[x1][v1*noOfCatAtts_+x2][v2*noOfClasses_+y]; }
//
//private:
//  const std::vector<std::vector<InstanceCount> >* subDist_;
//  const unsigned int noOfClasses_;
//  const unsigned int noOfCatAtts_;
//};
//
//class XXYSubDist2 {
//public:
//  XXYSubDist2(std::vector<std::vector<InstanceCount> >* subDist, const unsigned int noOfClasses,const unsigned int noOfCatAtts) : subDist_(subDist), noOfClasses_(noOfClasses),noOfCatAtts_(noOfCatAtts) {}
//
//  inline std::vector<InstanceCount>* getXYSubDist(const CategoricalAttribute x, const CatValue v) const { return &subDist_[x][v*noOfCatAtts_]; }
//  inline InstanceCount getCount(const CategoricalAttribute x1, const CatValue v1, const CategoricalAttribute x2, const CatValue v2, const CatValue y) const { return subDist_[x1][v1*noOfCatAtts_+x2][v2*noOfClasses_+y]; }
//  inline void incCount(const CategoricalAttribute x1, const CatValue v1, const CategoricalAttribute x2, const CatValue v2, const CatValue y) { ++subDist_[x1][v1*noOfCatAtts_+x2][v2*noOfClasses_+y]; }
//
//  std::vector<std::vector<InstanceCount> >* subDist_;
//  const unsigned int noOfClasses_;
//  const unsigned int noOfCatAtts_;
//};
//

class xxxyDist3 {
public:
	xxxyDist3();
	xxxyDist3(InstanceStream& stream);
	~xxxyDist3(void);

	void reset(InstanceStream& stream);

	void update(const instance& i);

	void clear();

//	// p(x1=v1, x2=v2, x3=v3, Y=y) unsmoothed
//	inline double rawJointP(CategoricalAttribute x1, CatValue v1,
//			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
//			CatValue v3, CatValue y) const {
//		return (*constRef(x1, v1, x2, v2, x3, v3, y))
//				/ (xxyCounts.xyCounts.count);
//	}
//
//	// p(x1=v1, x2=v2, x3=v3, Y=y) using M-estimate
//	inline double jointP(CategoricalAttribute x1, CatValue v1,
//			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
//			CatValue v3, CatValue y) const {
//		return (*constRef(x1, v1, x2, v2, x3, v3, y)
//				+ M
//						/ (getNoValues(x1) * getNoValues(x2) * getNoValues(x3)
//								* noClasses_)) / (xxyCounts.xyCounts.count + M);
//	}
//	// p(x1=v1, x2=v2, x3=v3) using M-estimate
//	inline double jointP(CategoricalAttribute x1, CatValue v1,
//			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
//			CatValue v3) const {
//		return (getCount(x1, v1, x2, v2, x3, v3)
//				+ M / (getNoValues(x1) * getNoValues(x2) * getNoValues(x3)))
//				/ (xxyCounts.xyCounts.count + M);
//
//	}
//// p(x1=v1|Y=y, x2=v2, x3=v3) using M-estimate
//	inline double p(CategoricalAttribute x1, CatValue v1,
//			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
//			CatValue v3, CatValue y) const {
//
//		return (*constRef(x1, v1, x2, v2, x3, v3, y) + M / getNoValues(x1))
//				/ (xxyCounts.getCount(order_[x2], v2, order_[x3], v3, y) + M);
//	}
//
//	// p(x1=v1, x2=v2, x3=v3, Y=y)
//	inline InstanceCount getCount(CategoricalAttribute x1, CatValue v1,
//			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
//			CatValue v3, CatValue y) const {
//		return *constRef(x1, v1, x2, v2, x3, v3, y);
//	}
//
//	// count for instances x1=v1, x2=v2,x3=v3
//	inline InstanceCount getCount(CategoricalAttribute x1, CatValue v1,
//			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
//			CatValue v3) const {
//		InstanceCount c = 0;
//
//		for (CatValue y = 0; y < noClasses_; y++) {
//			c += getCount(x1, v1, x2, v2, x3, v3, y);
//		}
//		return c;
//	}

	inline unsigned int getNoCatAtts() const {
		return noCatAtts_;
	}

	inline unsigned int getNoValues(const CategoricalAttribute a) const {
		return metaData_->getNoValues(order_[a]);
	}

	inline unsigned int getNoClasses() const {
		return noClasses_;
	}

	inline std::vector<std::vector<InstanceCount> >* getXXYSubDist(
			CategoricalAttribute x1, CatValue v1) {
		return &xxxyCount[x1][v1 * x1];
	}

	inline std::vector<std::vector<InstanceCount> >* getXXYSubDistRest(
			CategoricalAttribute x1, CatValue v1) {
		return &xxXyCount[x1][v1 * x1];
	}

//	inline std::vector<InstanceCount>* getXYSubDist(CategoricalAttribute x1,
//			CatValue v1, CategoricalAttribute x2, CatValue v2) {
//		return &xxxyCount[x1][v1 * x1 + x2][v2 * noCatAtts_];
//	}

	void setOrder(std::vector<CategoricalAttribute> &order);
	void setNoSelectedCatAtts(unsigned int noSelectedCatAtts);

private:
	// count[X1=x1][X2=x2][X3=x3][Y=y]
	inline InstanceCount *ref(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
			CatValue v3, CatValue y) {
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

		if (!(x1 > x2 && x2 > x3))
			printf("sequence of the attributes is not correct.\n");

		assert(x1 > x2 && x2 >x3);
		//	assert(x1 > x2 );
		return &xxxyCount[x1][v1 * x1 + x2][v2 * x2 + x3][v3 * noClasses_ + y];
	}
	// count[X1=x1][X2=x2][X3=x3][Y=y]
	inline InstanceCount *refRest(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
			CatValue v3, CatValue y) {
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
		//assert(x1 > x2 && x2 >x3);
		assert(x1 > x2);
		return &xxXyCount[x1][v1 * x1 + x2][v2 * noUnSelectedCatAtts_ + x3][v3
				* noClasses_ + y];
	}
//	// count[X1=x1][X2=x2][X3=x3]
//	inline std::vector<instanceCount> *xxxref(CategoricalAttribute x1, CatValue v1,
//			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3, CatValue v3) {
//		assert(x1 < x2 && x2 < x3);
//
//		return &count[x1][v1][x2][v2][x3][v3];
//	}

//// count[X1=x1][X2=x2][X3=x3][Y=y]
//	inline const InstanceCount *constRef(CategoricalAttribute x1, CatValue v1,
//			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
//			CatValue v3, CatValue y) const {
////		if (x2 > x1) {
////			CatValue t = x1;
////			x1 = x2;
////			x2 = t;
////			t = v1;
////			v1 = v2;
////			v2 = t;
////		}
////		if (x3 > x2) {
////			CatValue t = x2;
////			x2 = x3;
////			x3 = t;
////			t = v2;
////			v2 = v3;
////			v3 = t;
////		}
////		if (x2 > x1) {
////			CatValue t = x1;
////			x1 = x2;
////			x2 = t;
////			t = v1;
////			v1 = v2;
////			v2 = t;
////		}
//		assert(x1 > x2 && x2 > x3);
////		assert(x1 > x2 );
//		return &xxxyCount[x1][v1 * x1 + x2][v2 * x2 + x3][v3 * noClasses_ + y];
//	}

public:
	xxyDist xxyCounts;

private:
	unsigned int noCatAtts_; ///< the number of categorical CategoricalAttributes.
	unsigned int noClasses_;  ///< the number of classes

	unsigned int noSelectedCatAtts_; ///< the number of categorical CategoricalAttributes.
	unsigned int noUnSelectedCatAtts_; ///< the number of categorical CategoricalAttributes.

	InstanceStream::MetaData* metaData_;

	std::vector<CategoricalAttribute> order_;

	// a four-dimensional semiflattened representation of count[X1=x1][X2=x2][X3=x3][Y=y] storing only X1 > X2 > X3
	// outer vector is indexed by X1
	// second vector is indexed by x1*X2
	// third vector is indexed by x2*X3
	// inner vector is indexed by x3*y
	std::vector<std::vector<std::vector<std::vector<InstanceCount> > > > xxxyCount;
	std::vector<std::vector<std::vector<std::vector<InstanceCount> > > > xxXyCount;

};
