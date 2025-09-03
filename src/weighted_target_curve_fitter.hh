////////////////////////////////////////////////////////////////////////////////
// weighted_target_curve_fitter.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implementation of a weighted pointwise distance between two curves for inverse design
*/
////////////////////////////////////////////////////////////////////////////////
#ifndef WEIGHTEDTARGETCURVEFITTER_HH
#define WEIGHTEDTARGETCURVEFITTER_HH

#include "Tencer.hh"
#include "helpers.hh"


template<typename Real_>
struct WeightedTargetCurveFitter{
    using VecX   = VecX_T<Real>;
    using Vec3   = Vec3_T<Real>;
    using Mat3   = Mat3_T<Real>;
    using CSCMat_T = CSCMatrix<SuiteSparse_long, Real_>;

    ElasticRod target_rod;
    VecX r2;

    // Constructor
    WeightedTargetCurveFitter(const PeriodicRod &target, VecX &rad) : target_rod(target.rod), r2(rad) {
        if ((size_t)r2.size() != target.numVertices()) throw std::runtime_error("Size mismatch between radii and target rod (periodic rod)!");
        r2 = r2.cwiseProduct(r2);
    }

    WeightedTargetCurveFitter(const ElasticRod &target, VecX &rad) : target_rod(target), r2(rad) {
        if ((size_t)r2.size() != target_rod.numVertices()){
            throw std::runtime_error("Size mismatch between radii and target rod (elastic rod)!");
        } 
        r2 = r2.cwiseProduct(r2);
    }

    // Copy constructor
    template<typename Real_2>
    WeightedTargetCurveFitter(const WeightedTargetCurveFitter<Real_2>& tcf) : target_rod(tcf.target_rod), r2(tcf.r2) {}

    // affectation
    template<typename Real2>
    WeightedTargetCurveFitter<Real_>& operator=(const WeightedTargetCurveFitter<Real2> &tcf) { 
        target_rod = tcf.target_rod;
        r2 = tcf.r2;
        return *this;
    }



    // Energy = weighted distance^2 between two rods
    Real_ energy(const ElasticRod_T<Real_> &rod, size_t num_vertices) const { 
        VecX_T<Real_> diff = rod.getDoFs().head(3*num_vertices) - target_rod.getDoFs().head(3*num_vertices);
        VecX_T<Real_> d2 = vector_norms(diff, num_vertices); 
        d2 = d2.cwiseQuotient(r2);
        return d2.sum();
    }

    Real_ energy(const PeriodicRod_T<Real_> &rod) const { 
        return energy(rod.rod, rod.numVertices());
    }

    Real_ energy(const ElasticRod_T<Real_> &rod) const { 
        return energy(rod, rod.numVertices());
    }

    // Gradient with respect to rod defo variables
    VecX_T<Real_> grad(const ElasticRod_T<Real_> &rod, size_t num_vertices, size_t num_defo_vars) const {
        VecX_T<Real_> diff = VecX_T<Real_>::Zero(num_defo_vars);
        VecX_T<Real_> d = rod.getDoFs().head(3*num_vertices) - target_rod.getDoFs().head(3*num_vertices);
        diff.head(3 * num_vertices) = 2 * d.cwiseQuotient(repeat_interleave(r2));
        return diff;
    }

    VecX_T<Real_> grad(const PeriodicRod_T<Real_> &rod) const {
        return grad(rod.rod, rod.numVertices(),rod.numDoF());
    }

    VecX_T<Real_> grad(const ElasticRod_T<Real_> &rod) const {
        return grad(rod, rod.numVertices(),rod.numDoF());
    }


    /*
     * Used in multi-rod tencer optimization
     * start_index : first defo var index of the rod in result and delta_x
     */ 
    void apply_hessian(size_t num_vertices,const VecX_T<Real_> &delta_x, VecX_T<Real_> &result, Real weight, size_t start_index) const{
        result.segment(start_index, 3*num_vertices) = weight * 2 * delta_x.segment(start_index, 3*num_vertices).cwiseQuotient(repeat_interleave(r2));
    }
       

    void apply_hessian(const PeriodicRod_T<Real_> &rod,const VecX_T<Real_> &delta_x, VecX_T<Real_> &result, Real weight, size_t start_index) const{
        return apply_hessian(rod.numVertices(),delta_x,result,weight,start_index);
    }

    void apply_hessian(const ElasticRod_T<Real_> &rod,const VecX_T<Real_> &delta_x, VecX_T<Real_> &result, Real weight, size_t start_index) const{
        return apply_hessian(rod.numVertices(),delta_x,result,weight,start_index);
    }
    

};

template struct WeightedTargetCurveFitter<Real>;
template struct WeightedTargetCurveFitter<ADReal>;



#endif /* end of include guard: WEIGHTEDTARGETCURVEFITTER_HH */