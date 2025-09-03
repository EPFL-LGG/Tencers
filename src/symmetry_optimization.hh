#ifndef SYMMETRYOPTIMIZATION_HH
#define SYMMETRYOPTIMIZATION_HH

#include <DesignOptimization/OptimizationObjectWithLinearChangeOfVariable.hh>
#include "Tencer.hh"
#include "tencer_optimization.hh"

/*
 * In this class, we handle 2 kinds of changes of variables for symmetries : 
 * - linear constraints (for stiffness equality constraints) : transformation matrix . reduced_vars = unreduced_vars
 * - affine constraints (for spring endpoint positions symmetry constraints) : transformation matrix . reduced_vars + length_offset = unreduced_vars
 */


struct TencerOptimizationSymmetries : public OptimizationObjectWithLinearChangeOfVariable<Tencer_T,OptEnergyType> {

    using Tencer = Tencer_T<Real>;
    using ADTencer = Tencer_T<ADReal>;
    using KnotEnergyType = Tencer::KnotEnergyType;
    using CSCMat = CSCMatrix<SuiteSparse_long, Real>;

    // Type of transformation : linear or affine
    enum class TransformationType {Linear, Affine};
    
    // Constructors
    TencerOptimizationSymmetries(CSCMat &v,Tencer &knot, const NewtonOptimizerOptions &eopts, const std::vector<size_t> &fixedVars, std::vector<ElasticRod> &open_target_rods, std::vector<PeriodicRod> &closed_target_rods, std::vector<VecX> &radii, VecX weight = {}, const VecX &lengths_shift = {}, Real hessian_shift = 1e-6, PredictionOrder po = PredictionOrder::One);

    // Override objective and derivatives
    virtual Real J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType = OptEnergyType::Full) override;

    virtual Eigen::VectorXd gradp_J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType = OptEnergyType::Full) override;

    virtual Eigen::VectorXd apply_hess(const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, Real coeff_J, OptEnergyType opt_eType = OptEnergyType::Full) override;

    // Design variables
    virtual const Eigen::VectorXd params() const override;

    // Optimizer new point callback (to evaluate a new set of parameters)
    virtual void newPt(const Eigen::VectorXd &params) override;

    // Affine transformation when length offsets
    VecX apply_transformation_affine(const VecX &so) const;

    // Update target rods after realignment
    void update_target_rods(std::vector<ElasticRod> &open_target_rods,std::vector<PeriodicRod> &closed_target_rods,std::vector<VecX> &radii);
    
    // Get variables bounds
    std::vector<Real> get_vars_lower_bounds() const;
    std::vector<Real> get_vars_upper_bounds() const;


    // Attributes
    std::vector<int> symmetric_groups_reps; // Representatives for each symmetry group
    VecX length_offset; // Length offsets for sliding nodes symmetries (affine transformation)
    TransformationType transformation_type; // Linear or affine transformation

};


#endif