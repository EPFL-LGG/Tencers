#ifndef HELPERS_HH
#define HELPERS_HH
#include <ElasticRods/PeriodicRod.hh>

using VecX = VecX_T<Real>;
template<typename Real_> using MatX3_T = Eigen::Matrix<Real_, Eigen::Dynamic, 3>;

/*
* Consecutively repeat 3 times elements of a vector
* Ex : repeat_interleave([x,y,z]) = [x,x,x,y,y,y,z,z,z]
*/
template<typename Real_>
VecX_T<Real_> repeat_interleave(const VecX_T<Real_> &v) {
    MatX3_T<Real_> v_rep = v.template replicate<1,3>();
    Mat3X_T<Real_> v_rep_t = v_rep.transpose();
    return VecX_T<Real_>::Map(v_rep_t.data(),3*v.size());
}

/*
* Squared norm of all vectors
* Ex : vector_norms([x1,y1,z1,x2,y2,z2]) = [x1^2+y1^2+z1^2, x2^2+y2^2+z2^2]
*/
template<typename Real_>
VecX_T<Real_> vector_norms(VecX_T<Real_> &diff, size_t num_vertices){
    VecX_T<Real_> diff2 = diff.cwiseProduct(diff);
    return Mat3X_T<Real_>::Map(diff2.data(),3, num_vertices).colwise().sum();
}

/*
 * Minimize twist wrt theta on a periodic rod
 * (only twist, bending is not affected)
 * Rod starting and ending points can rotate freely to remove unnecessary twist
 * (from RodLinkage.cc)
 */
inline void minimize_twist(PeriodicRod &pr, bool verbose = false){
    
    ElasticRod &rod = pr.rod;
    const size_t ne = rod.numEdges();
    auto pts = rod.deformedPoints();
    auto ths = rod.thetas();

    // First, remove any unnecessary twist stored in the rod by rotating the second endpoint
    // by an integer multiple of 2PI (leaving d2 unchanged).
    Real rodRefTwist = 0;
    const auto &dc = rod.deformedConfiguration();
    for (size_t j = 1; j < ne; ++j)
        rodRefTwist += dc.referenceTwist[j];
    const size_t lastEdge = ne - 1;
    Real desiredTheta = ths[0] - rodRefTwist;
    
    while (ths[lastEdge] - desiredTheta >  M_PI) ths[lastEdge] -= 2 * M_PI;
    while (ths[lastEdge] - desiredTheta < -M_PI) ths[lastEdge] += 2 * M_PI;

    if (verbose) {
        std::cout << "rodRefTwist: "         << rodRefTwist        << std::endl;
        std::cout << "desiredTheta: "        << desiredTheta       << std::endl;
        std::cout << "old last edge theta: " << dc.theta(lastEdge) << std::endl;
        std::cout << "new last edge theta: " << ths[lastEdge]      << std::endl;
    }
    
    rod.setDeformedConfiguration(pts, ths);

    auto H = rod.hessThetaEnergyTwist();
    auto g = rod.gradEnergyTwist();
    std::vector<Real> rhs(ne);
    for (size_t j = 0; j < ne; ++j)
        rhs[j] = -g.gradTheta(j);

    H.fixVariable(0, 0); // lock rotation in the first edge
    auto thetaStep = H.solve(rhs);

    for (size_t j = 0; j < ths.size(); ++j)
        ths[j] += thetaStep[j];

    rod.setDeformedConfiguration(pts, ths);
    pr.setTotalOpeningAngle(ths[ne-1] - ths[0]);  // set the total opening angle of the input PeriodicRod s.t. it is consistent with the twist of the underlying ElasticRod
}

/*
 * Minimize twisting energy wrt theta while keeping
 * the first and the last edges of the underlying ElasticRod fixed
 * (from ElasticKnots/SlidingProblem.cc)
 */
inline void spread_twist_preserving_link(PeriodicRod &pr, bool verbose) { 

    if (verbose) {
        std::cout << "Initial total twist: " << pr.twist() << std::endl; 
        std::cout << "Initial twisting energy: " << pr.energyTwist() << std::endl; 
    }

    ElasticRod &rod = pr.rod;
    const size_t ne = rod.numEdges();
    auto pts = rod.deformedPoints();
    auto ths = rod.thetas();

    auto H = rod.hessThetaEnergyTwist();
    auto g = rod.gradEnergyTwist();
    std::vector<Real> rhs(ne);
    for (size_t j = 0; j < ne; ++j)
        rhs[j] = -g.gradTheta(j);

    H.fixVariable(0,                   0.0);
    H.fixVariable(pr.rod.numEdges()-1, 0.0);
    auto thetaStep = H.solve(rhs);

    for (size_t j = 0; j < ths.size(); ++j)
        ths[j] += thetaStep[j];

    rod.setDeformedConfiguration(pts, ths);

    // As the first and last edges are fixed, the total opening angle is unchanged: 
    // no need to call pr.setTotalOpeningAngle(ths[ne-1] - ths[0])

    if (verbose) {
        std::cout << "Final total twist: " << pr.twist() << std::endl; 
        std::cout << "Final twisting energy: " << pr.energyTwist() << std::endl; 
    }
}


template<typename TIn, typename AllocIn, typename TOut, typename AllocOut>
static void castStdADVector(const std::vector<TIn, AllocIn> &in, std::vector<TOut, AllocOut> &out) {
    out.clear();
    out.reserve(in.size());
    for (const auto &val : in) out.push_back(autodiffCast<TOut>(val));
}

template<typename TIn, typename AllocIn, typename TOut, typename AllocOut>
static void castStdADVectorOfVectors(const std::vector<std::vector<TIn, AllocIn>> &in, std::vector<std::vector<TOut, AllocOut>> &out) {
    out.clear();
    out.reserve(in.size());
    for (const auto &vec : in) {
        std::vector<TOut, AllocOut> outVec;
        castStdADVector(vec, outVec);
        out.push_back(outVec);
    }
}

// Extend `result` from size (n x n) to (finalSize x finalSize) by adding empty columns/rows on the right/bottom
template<typename Real_>
inline void extendSparseMatrixSouthEast(CSCMatrix<SuiteSparse_long, Real_> &result, size_t finalSize) {
    assert(result.m == result.n && result.n <= (long)finalSize);
    const size_t additionalColumns = finalSize - result.n;
    result.m = result.n = finalSize;
    result.Ap.reserve(finalSize + 1);
    for (size_t i = 0; i < additionalColumns; i++)
        result.Ap.push_back(result.nz);
}

#endif