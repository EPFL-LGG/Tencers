#include "symmetry_optimization.hh"



/*********************************************************
 **************   SYMMETRY SPARSIFICATION   **************
 *********************************************************/

// Constructor
TencerOptimizationSymmetries::TencerOptimizationSymmetries(CSCMat &m,Tencer &knot, const NewtonOptimizerOptions &eopts, const std::vector<size_t> &fixedVars, std::vector<ElasticRod> &open_target_rods, std::vector<PeriodicRod> &closed_target_rods, std::vector<VecX> &radii, VecX weight, const VecX &lengths_shift, Real hessian_shift, PredictionOrder po) :
    OptimizationObjectWithLinearChangeOfVariable(knot,m,std::make_shared<TencerOptimization>(knot,eopts,fixedVars,open_target_rods,closed_target_rods,radii, weight, hessian_shift,po)),
    length_offset(lengths_shift),
    transformation_type(TransformationType::Linear)
{

    // Transformation type : affine or linear
    if (length_offset.size() > 0) transformation_type = TransformationType::Affine;

    // Symmetry groups representatives
    for (auto i=0; i < transformation_matrix.n; ++i){
        symmetric_groups_reps.push_back(transformation_matrix.Ai[transformation_matrix.Ap[i]]);
    }

}

// Objective and derivatives

Real TencerOptimizationSymmetries::J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType) {
    if (transformation_type == TransformationType::Linear)
        return OptimizationObjectWithLinearChangeOfVariable::J(params, opt_eType); 
    return optimization_object->J(apply_transformation_affine(params), opt_eType); 
}

Eigen::VectorXd TencerOptimizationSymmetries::gradp_J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType) { 
    if (transformation_type == TransformationType::Linear)
        return OptimizationObjectWithLinearChangeOfVariable::gradp_J(params, opt_eType); 
    return apply_transformation_transpose(optimization_object->gradp_J(apply_transformation_affine(params), opt_eType)); 
}

Eigen::VectorXd TencerOptimizationSymmetries::apply_hess(const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, Real coeff_J, OptEnergyType opt_eType){ 
    if (transformation_type == TransformationType::Linear)
        return OptimizationObjectWithLinearChangeOfVariable::apply_hess(params, delta_p, coeff_J, opt_eType); 
    return apply_transformation_transpose(optimization_object->apply_hess(apply_transformation_affine(params), apply_transformation(delta_p), coeff_J, opt_eType)); 
}

// Design variables
const VecX TencerOptimizationSymmetries::params() const{
    VecX res = VecX::Zero(numParams());
    VecX p = optimization_object->params();
    if (transformation_type == TransformationType::Linear){
        for (size_t i = 0; i < symmetric_groups_reps.size(); ++i){
            // all linear constraints are equality constraints y = x
            res[i] = p[symmetric_groups_reps[i]];
        }
    }
    else{
        for (size_t i = 0; i < symmetric_groups_reps.size(); ++i){
            // all affine constraints are of the form y = ax + b
            auto idx = symmetric_groups_reps[i];
            res[i] = (p[idx] - length_offset[idx])/transformation_matrix.Ax[transformation_matrix.Ap[i]];
        }
    }
    return res;
}

// Optimizer new point callback (to evaluate a new set of parameters)
void TencerOptimizationSymmetries::newPt(const Eigen::VectorXd &params)  {
    if (transformation_type == TransformationType::Linear)
        OptimizationObjectWithLinearChangeOfVariable::newPt(params);
    else optimization_object->newPt(apply_transformation_affine(params));
}

// Update target rods after realignment
void TencerOptimizationSymmetries::update_target_rods(std::vector<ElasticRod> &open_target_rods,std::vector<PeriodicRod> &closed_target_rods,std::vector<VecX> &radii){
    std::dynamic_pointer_cast<TencerOptimization>(optimization_object)->update_target_rods(open_target_rods,closed_target_rods,radii);
}

// Affine transformation when length offsets
VecX TencerOptimizationSymmetries::apply_transformation_affine(const VecX &so) const{
    VecX res = apply_transformation(so) + length_offset;
    return res;
}




// Get variables bounds

std::vector<Real> TencerOptimizationSymmetries::get_vars_lower_bounds() const { 
        std::vector<Real> all_bounds = m_object.get_rest_vars_lower_bounds(); 
        std::vector<Real> res = {};
        for (auto g : symmetric_groups_reps){
            res.push_back(all_bounds[g]);
        }
        return res;
    }

std::vector<Real> TencerOptimizationSymmetries::get_vars_upper_bounds() const { 
    std::vector<Real> all_bounds = m_object.get_rest_vars_upper_bounds(); 
    std::vector<Real> res = {};
    for (auto g : symmetric_groups_reps){
        res.push_back(all_bounds[g]);
    }
    return res; 
}