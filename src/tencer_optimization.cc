#include "tencer_optimization.hh"

/******************************************************
 **************   TENCER OPTIMIZATION   ***************
 ******************************************************/


TencerOptimization::TencerOptimization(Tencer &knot, const NewtonOptimizerOptions &eopts, const std::vector<size_t> &fixedVars, std::vector<ElasticRod> &open_target_rods, std::vector<PeriodicRod> &closed_target_rods, std::vector<VecX> &radii, VecX weight, Real hessian_shift, PredictionOrder po) : 
EquilibriumOptimizationObject(knot, eopts, fixedVars,hessian_shift,po), hess_shift(hessian_shift){
    using OET = OptEnergyType;
    using WTFO = WeightedTargetFittingDOOT;

    r = 0.5; //default value 1/2 for norm parameter
    if (weight.size() == 0){
        weight = VecX::Zero(knot.numRestVars());
    }
    sparsification_weight = weight;
    is_zero_weight = (weight.array() < 1e-6).matrix().cast<double>();
    is_non_zero_weight = (weight.array() > 1e-6).matrix().cast<double>();
    if (knot.get_rest_vars_type() == Spring_T<Real>::RestVarsType::SpringAnchors || knot.get_rest_vars_type() == Spring_T<Real>::RestVarsType::StiffnessAndSpringAnchors) optimize_spring_positions = true;
    else optimize_spring_positions = false;

    objective.add("TargetCurveFitting", OET::CurveFitting, std::make_shared<WTFO>(m_linesearch_object, open_target_rods, closed_target_rods, radii), 1.0);
}


TencerOptimization::TencerOptimization(TencerOptimization &sp) : 
EquilibriumOptimizationObject(sp.m_object, sp.getEquilibriumOptions(), sp.m_fixedVars,sp.hess_shift,sp.prediction_order) {
    for (auto t:sp.objective.terms){
        objective.add(t.name,t.type, t.term,t.term->getWeight());
    }
    r = sp.r; //default value 1/2 for norm parameter
    sparsification_weight = sp.sparsification_weight;
    is_zero_weight = sp.is_zero_weight;
    is_non_zero_weight = sp.is_non_zero_weight;
    optimize_spring_positions = sp.optimize_spring_positions;
}

Real TencerOptimization::J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType){
    VecX stiffness = is_non_zero_weight.cwiseProduct(params.array().pow(1/r).matrix()) + is_zero_weight.cwiseProduct(params);
    m_updateEquilibria(stiffness);
    if (!m_equilibrium_solve_successful) return std::numeric_limits<Real>::max();
    Real v = objective.value(opt_eType) + sparsification_weight.dot(params);
    return v;
}

Eigen::VectorXd TencerOptimization::gradp_J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType) {
    // update adjoint state
    VecX stiffness = is_non_zero_weight.cwiseProduct(params.array().pow(1/r).matrix()) + is_zero_weight.cwiseProduct(params);

    // Change of variable
    VecX grad_obj_K = apply_change_variable(params, EquilibriumOptimizationObject::gradp_J(stiffness,opt_eType));

    return grad_obj_K + sparsification_weight;
}

Eigen::VectorXd TencerOptimization::apply_hess(const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, Real coeff_J, OptEnergyType opt_eType){
    VecX result;
    VecX stiffness = is_non_zero_weight.cwiseProduct(params.array().pow(1/r).matrix()) + is_zero_weight.cwiseProduct(params);
    result = apply_change_variable(params,delta_p);
    result = EquilibriumOptimizationObject::apply_hess(stiffness,result,coeff_J,opt_eType);
    result = apply_change_variable(params,result);
    result += apply_change_variable_second_order(params, delta_p, dJ_dp(opt_eType));
    return result;
}

// Gradient without the need to recompute adjoint state, used for the change of variable in the hessian formula
Eigen::VectorXd TencerOptimization::dJ_dp(OptEnergyType opt_eType){
    const size_t nd = m_linesearch_object.numDefoVars();
    const size_t np = numParams();
    Eigen::VectorXd y_padded(nd + np);
    y_padded.head(nd) = this->objective.adjointState();
    y_padded.tail(np).setZero();
    return this->objective.grad_p(opt_eType) - m_linesearch_object.apply_hessian_defoVarsIn_restVarsOut(y_padded).tail(np);
}

VecX TencerOptimization::apply_change_variable(const VecX &params, const VecX &v){
    // Here params are already stiffness to the power of r
    return 1 / r * params.array().pow(1/r-1).matrix().cwiseProduct(v).cwiseProduct(is_non_zero_weight) + v.cwiseProduct(is_zero_weight);
}

VecX TencerOptimization::apply_change_variable_second_order(const VecX &params, const VecX &delta_params, const VecX &v){
    // Here params are already stiffness to the power of r
    return 1 / r * (1/r - 1) * params.array().pow(1/r-2).matrix().cwiseProduct(v).cwiseProduct(delta_params).cwiseProduct(is_non_zero_weight);
}

void TencerOptimization::newPt(const Eigen::VectorXd &params){
    VecX stiffness = is_non_zero_weight.cwiseProduct(params.array().pow(1/r).matrix()) + is_zero_weight.cwiseProduct(params);
    std::cout << "newPt at dist " << (m_object.getRestVars() - stiffness).norm() << std::endl;
    m_updateAdjointState(stiffness,OptEnergyType::Full); // update the adjoint state as well as the equilibrium, since we'll probably be needing gradients at this point.
    commitLinesearchObject();
}

void TencerOptimization::update_target_rods(std::vector<ElasticRod> &open_target_rods,std::vector<PeriodicRod> &closed_target_rods,std::vector<VecX> &radii){
    objective.remove("TargetCurveFitting"); // remove already throws an error if the term does not exist
    objective.add("TargetCurveFitting", OptEnergyType::CurveFitting, std::make_shared<WeightedTargetFittingDOOT>(m_linesearch_object, open_target_rods, closed_target_rods, radii), 1.0);
}



bool TencerOptimization::m_updateEquilibria(const Eigen::Ref<const Eigen::VectorXd> &newParams){
    BENCHMARK_SCOPED_TIMER_SECTION timer("m_updateEquilibria");
    if ( (size_t) newParams.size() != numParams()) throw std::runtime_error("Parameter vector size mismatch");

    if ((m_linesearch_object.getRestVars() - newParams).norm() < 1e-16) return false;
    m_linesearch_object.set(m_object);


    const Eigen::VectorXd currParams = m_object.getRestVars();
    Eigen::VectorXd delta_p = newParams - currParams;


    if (delta_p.squaredNorm() == 0) { // returning to linesearch start; no prediction/Hessian factorization necessary
        if (optimize_spring_positions) getEquilibriumOptimizer().get_problem().sparsityPatternFactorizationUpToDate(false);
        m_forceEquilibriumUpdate();
        return true;
    }

    // Apply the new design parameters and measure the energy with the 0^th order prediction
    // (i.e. at the currently committed equilibrium).
    // We will only replace this equilibrium if the higher-order predictions achieve a lower energy.
    m_linesearch_object.setRestVars(newParams);
    if (optimize_spring_positions) getEquilibriumOptimizer().get_problem().sparsityPatternFactorizationUpToDate(false);
    Real bestEnergy = m_linesearch_object.energy();
    Eigen::VectorXd curr_x = m_object.getDefoVars();
    Eigen::VectorXd best_x = curr_x;


    if (prediction_order > PredictionOrder::Zero) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("Predict equilibrium");
        // Return to using the Hessian for the last committed linkage
        // (i.e. for the equilibrium stored in m_flat and m_deployed).
        auto &opt_object = getEquilibriumOptimizer();
        if (!opt_object.solver().hasStashedFactorization()){
            throw std::runtime_error("Factorization was not stashed... was commitLinesearchLinkage() called?");
        }
        opt_object.solver().swapStashedFactorization();

        {
            // Solve for equilibrium perturbation corresponding to delta_p:
            //      [H][delta x] = [-d2E/dxdp delta_p]
            //                     \_________________/
            //                              b
            const size_t np = numParams(), nd = m_object.numDefoVars();
            VecX_T<Real> neg_deltap_padded(nd + np);
            neg_deltap_padded.setZero();
            neg_deltap_padded.tail(np) = -delta_p;



            VecX_T<Real> neg_d2E_dxdp_deltap = m_object.apply_hessian_defoVarsOut_restVarsIn(neg_deltap_padded).head(nd);
            m_committed_ws->getFreeComponentInPlace(neg_d2E_dxdp_deltap); // Enforce any active bound constraints (equilibrium optimizer already modified the Hessian accordingly)
            m_delta_x = opt_object.extractFullSolution(opt_object.solver().solve(opt_object.removeFixedEntries(neg_d2E_dxdp_deltap)));
            m_committed_ws->validateStep(m_delta_x); // Debugging: ensure bound constraints are respected perfectly.

            // Evaluate the energy at the 1st order-predicted equilibrium
            {
                auto first_order_x = (curr_x + m_delta_x).eval();
                m_linesearch_object.setDefoVars(first_order_x);
                Real energy1stOrder = m_linesearch_object.energy();
                if (energy1stOrder < bestEnergy) { std::cout << " used first order prediction, energy reduction " << bestEnergy - energy1stOrder << std::endl; bestEnergy = energy1stOrder; best_x = first_order_x; } else { this->m_linesearch_object.setDefoVars(best_x); }
            }

            if (prediction_order > PredictionOrder::One) {
                // Solve for perturbation of equilibrium perturbation corresponding to delta_p:
                //      H (delta_p^T d2x/dp^2 delta_p) = -(d3E/dx3 delta_x + d3E/dx2dp delta_p) delta_x -(d3E/dxdpdx delta_x + d3E/dxdpdp delta_p) delta_p
                //                                     = -directional_derivative_delta_dxp(d2E/dx2 delta_x + d2E/dxdp delta_p)  (Directional derivative computed with autodiff)
                m_diff_object.set(m_object);

                Eigen::VectorXd neg_d3E_delta_x;
                {
                    VecX_T<Real> delta_xp(nd + np);
                    delta_xp << m_delta_x, delta_p;

                    // inject equilibrium and design parameter perturbation
                    VecX_T<ADReal> ad_xp =m_object.getVars(VariableMask::All); 
                    for (size_t i = 0; i < nd + np; ++i) ad_xp[i].derivatives()[0] = delta_xp[i];
                    m_diff_object.setVars(ad_xp,VariableMask::All); 

                    neg_d3E_delta_x = -extractDirectionalDerivative(m_diff_object.apply_hessian(delta_xp)).head(nd);
                }

                m_committed_ws->getFreeComponentInPlace(neg_d3E_delta_x); // Enforce any active bound constraints.
                // Eigen::VectorXd m_delta_delta_x = opt_object.solver().solve(neg_d3E_delta_x);
                Eigen::VectorXd m_delta_delta_x = opt_object.extractFullSolution(opt_object.solver().solve(opt_object.removeFixedEntries(neg_d3E_delta_x)));
                m_committed_ws->validateStep(m_delta_delta_x); // Debugging: ensure bound constraints are respected perfectly.

                // Evaluate the energy at the 2nd order-predicted equilibrium, roll back to previous best if energy is higher.
                {
                    Eigen::VectorXd m_second_order_x = (curr_x + m_delta_x + 0.5 * m_delta_delta_x).eval();
                    m_linesearch_object.setDefoVars(m_second_order_x);
                    Real energy2ndOrder = m_linesearch_object.energy();
                    if (energy2ndOrder < bestEnergy) { std::cout << " used second order prediction, energy reduction " << bestEnergy - energy2ndOrder << std::endl; bestEnergy = energy2ndOrder; best_x = m_second_order_x;} else { this->m_linesearch_object.setDefoVars(best_x); }
                }
            }
        }

        // Return to using the primary factorization, storing the committed
        // linkages' factorizations back in the stash for later use.
        opt_object.solver().swapStashedFactorization();
    }
    this->m_forceEquilibriumUpdate();

    return true;
}