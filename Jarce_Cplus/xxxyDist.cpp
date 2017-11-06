#include "xxxyDist.h"

xxxyDist::xxxyDist() {

}
xxxyDist::xxxyDist(InstanceStream& stream) {
	reset(stream);

	// pass through the stream updating the counts incrementally
	stream.rewind();

	instance i;

	while (stream.advance(i)) {
		update(i);
	}

}

void xxxyDist::reset(InstanceStream& stream) {

	metaData_ = stream.getMetaData();

	xxyCounts.reset(stream);

	noCatAtts_ = metaData_->getNoCatAtts();
	noClasses_ = metaData_->getNoClasses();

	//out vector
	count.resize(noCatAtts_);
	for (CategoricalAttribute x1 = 2; x1 < noCatAtts_; x1++) {

		//second vector
		count[x1].resize(metaData_->getNoValues(x1) * x1);
		for (CatValue v1 = 0; v1 < metaData_->getNoValues(x1); v1++) {
			for (CategoricalAttribute x2 = 1; x2 < x1; x2++) {

				//third vector
				count[x1][v1 * x1 + x2].resize(
						metaData_->getNoValues(x2) * x2);
				for (CatValue v2 = 0; v2 < metaData_->getNoValues(x2);
						v2++) {
					for (CategoricalAttribute x3 = 0; x3 < x2; x3++) {

						//inner vector
						count[x1][v1 * x1 + x2][v2 * x2 + x3].assign(
								metaData_->getNoValues(x3) * noClasses_,
								0);
					}
				}
			}
		}
	}
}

xxxyDist::~xxxyDist(void) {

}

void xxxyDist::update(const instance &i) {
	xxyCounts.update(i);

	const CatValue theClass = i.getClass();

	for (CategoricalAttribute x1 = 2; x1 < noCatAtts_; x1++) {
		const CatValue v1 = i.getCatVal(x1);

		XXYSubDist xxySubDist(getXXYSubDist(x1, v1), noClasses_ );

		for (CategoricalAttribute x2 = 1; x2 < x1; x2++) {
			const CatValue v2 = i.getCatVal(x2);

//		    XYSubDist xySubDist(getXYSubDist(x1, v1,x2,v2), noClasses_);

			for (CategoricalAttribute x3 = 0; x3 < x2; x3++) {
				const CatValue v3 = i.getCatVal(x3);

//				xySubDist.incCount(x3,v3,theClass);
				 xxySubDist.incCount(x2,v2,x3, v3, theClass);

				 assert(*ref(x1,v1,x2,v2,x3,v3,theClass) <= xxyCounts.xyCounts.count);
			}
		}
	}

}

void xxxyDist::clear(){
  count.clear();
  xxyCounts.clear();
}
