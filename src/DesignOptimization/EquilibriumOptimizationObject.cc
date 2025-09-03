#include "OptimizationObject.hh"
#include "EquilibriumOptimizationObject.hh"
#include "../TencerEquilibriumProblem.hh"

/*************************************************************
 ********************   CONSTRUCTOR   ************************
 *************************************************************/

template<template<typename> class Object_T, class EType>
EquilibriumOptimizationObject<Object_T,EType>::EquilibriumOptimizationObject(Object &object, const NewtonOptimizerOptions &eopts, const std::vector<size_t> &fixedVars,Real hessian_shift, PredictionOrder po) : 
    OptimizationObject<Object_T,EType>(object), 
    m_linesearch_object(object),  
    m_diff_object(object), 
    m_numParams(object.numRestVars()), 
    m_equilibrium_options(eopts) {

    // In inherited classes, the objective needs to be defined

    // fixed variables
    m_fixedVars.insert(std::end(m_fixedVars), std::begin(fixedVars), std::end(fixedVars));

    m_equilibrium_optimizer = get_equilibrium_optimizer(m_linesearch_object, m_fixedVars, m_equilibrium_options, nullptr,safe_numeric_limits<Real>::max(),1e-6,hessian_shift);
    prediction_order = po;

    // Ensure we start at an equilibrium (using the passed equilibrium solver options)
    m_forceEquilibriumUpdate();
    commitLinesearchObject();
}

/*************************************************************
 *************   OBJECTIVE AND DERIVATIVES   *****************
 *************************************************************/

// Objective value
template<template<typename> class Object_T, class EType>
Real EquilibriumOptimizationObject<Object_T,EType>::J(const Eigen::Ref<const Eigen::VectorXd> &params, EType opt_eType){

    // Solve equilibrium
    m_updateEquilibria(params);
    if (!m_equilibrium_solve_successful) return std::numeric_limits<Real>::max();
    return this->objective.value(opt_eType);
}

// Objective gradient
template<template<typename> class Object_T, class EType>
Eigen::VectorXd EquilibriumOptimizationObject<Object_T,EType>::gradp_J(const Eigen::Ref<const Eigen::VectorXd> &params, EType opt_eType){
    m_updateAdjointState(params, opt_eType);
    const size_t nd = m_linesearch_object.numDefoVars();
    const size_t np = numParams();
    Eigen::VectorXd y_padded(nd + np);
    y_padded.head(nd) = this->objective.adjointState();
    y_padded.tail(np).setZero();
    return this->objective.grad_p(opt_eType) - m_linesearch_object.apply_hessian_defoVarsIn_restVarsOut(y_padded).tail(np);
}

// Apply objective hessian
template<template<typename> class Object_T, class EType>
Eigen::VectorXd EquilibriumOptimizationObject<Object_T,EType>::apply_hess(const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, Real coeff_J, EType opt_eType){
    BENCHMARK_SCOPED_TIMER_SECTION timer("apply_hess_J");
    BENCHMARK_START_TIMER_SECTION("Preamble");
    const size_t np = numParams(), nd = m_linesearch_object.numDefoVars();
    if (size_t( params.size()) != np) throw std::runtime_error("Incorrect parameter vector size");
    if (size_t(delta_p.size()) != np) throw std::runtime_error("Incorrect delta parameter vector size");
    m_updateAdjointState(params, opt_eType);

    if (!m_autodiffObjectIsCurrent) {
        BENCHMARK_SCOPED_TIMER_SECTION timer2("Update autodiff");
        m_diff_object.set(m_linesearch_object);
        m_autodiffObjectIsCurrent = true;
    }

    auto &opt = getEquilibriumOptimizer();
    auto &H   = opt.solver();

    BENCHMARK_STOP_TIMER_SECTION("Preamble");

    VecX_T<Real> neg_deltap_padded(nd + np);
    neg_deltap_padded.head(nd).setZero();
    neg_deltap_padded.tail(np) = -delta_p;

    VecX_T<Real> delta_dJ_dxp;
    Eigen::VectorXd d3E_y;

     try {
        // Solve for state perturbation
        // d2E/dx2 dx*/dp = - d2E/dxdp
        // ==> d2E/dx2 delta_x = - d2E/dxdp delta_p
        // H delta x = [-d2E/dxdp delta_p]
        //             \_________________/
        //                     b
        {
            BENCHMARK_SCOPED_TIMER_SECTION timer2("solve delta x");
            VecX_T<Real> b = m_linesearch_object.apply_hessian_defoVarsOut_restVarsIn(neg_deltap_padded).head(nd);
            m_linesearch_ws->getFreeComponentInPlace(b); // Enforce any active bound constraints (equilibrium optimizer already modified the Hessian accordingly)
            m_delta_x = opt.extractFullSolution(H.solve(opt.removeFixedEntries(b)));
            m_linesearch_ws->validateStep(m_delta_x); // Debugging: ensure bound constraints are respected perfectly.
        }

        VecX_T<Real> delta_xp(nd + np);
        delta_xp << m_delta_x, delta_p;
        

        // Solve for adjoint state perturbation
        BENCHMARK_START_TIMER_SECTION("getDoFs and inject state");
        VecX_T<ADReal> ad_xp = m_linesearch_object.getVars(VariableMask::All);
        for (size_t i = 0; i < np + nd; ++i) ad_xp[i].derivatives()[0] = delta_xp[i];
        m_diff_object.setVars(ad_xp, VariableMask::All);
        BENCHMARK_STOP_TIMER_SECTION("getDoFs and inject state");

        BENCHMARK_START_TIMER_SECTION("delta_dJ_dxp");
        delta_dJ_dxp = this->objective.delta_grad(delta_xp, m_diff_object, opt_eType);
        BENCHMARK_STOP_TIMER_SECTION("delta_dJ_dxp");

        // Solve for adjoint state perturbation
        // d2E/dx2 dy/dp = d2J/dx2 dx*/dp + d2J/dxdp - (d3E/dx3 dx*/dp + d3E/dx2dp) y
        // H delta_y = [ d^2J/dxdxp delta_xp ] - [d3E/dx dx dxp delta_xp] y
        //             \__________________________________________________/
        //                                     b
        if (coeff_J != 0.0) {
            BENCHMARK_SCOPED_TIMER_SECTION timer2("solve delta y x");
            BENCHMARK_START_TIMER_SECTION("Hy");
            VecX_T<ADReal> y_padded(nd + np);
            y_padded.head(nd) = this->objective.adjointState();
            y_padded.tail(np).setZero();
            // Note: we need the "p" rows of d3E_y for evaluating the full Hessian matvec expressions below...
            VecX_T<ADReal> hxp = m_diff_object.apply_hessian_defoVars_restVarsOut(y_padded);
            d3E_y = extractDirectionalDerivative(hxp);
            BENCHMARK_STOP_TIMER_SECTION("Hy");

            BENCHMARK_START_TIMER_SECTION("KKT_solve");

            auto b = (delta_dJ_dxp.head(nd) - d3E_y.head(nd)).eval();
            m_linesearch_ws->getFreeComponentInPlace(b); // Enforce any active bound constraints
                                
            m_delta_y =  opt.extractFullSolution(opt.solver().solve(opt.removeFixedEntries(b)));

            m_linesearch_ws->validateStep(m_delta_y); // Debugging: ensure the adjoint state components corresponding to constrained variables remain zero.

            BENCHMARK_STOP_TIMER_SECTION("KKT_solve");
        }
    }
    catch (...) {
        std::cout << "HESSVEC FAIL!!!" << std::endl;
        m_delta_x    = VecX_T<Real>::Zero(nd     );
        m_delta_y    = VecX_T<Real>::Zero(nd     );
        d3E_y      = VecX_T<Real>::Zero(nd + np);
        delta_dJ_dxp = VecX_T<Real>::Zero(nd + np);
    }

    VecX_T<Real> result;
    result.setZero(np);
    // Accumulate the J hessian matvec
    {
        BENCHMARK_SCOPED_TIMER_SECTION timer3("evaluate hessian matvec");
        if (coeff_J != 0.0) {
            VecX_T<Real> delta_edofs(nd + np);
            delta_edofs.head(nd) = m_delta_y;
            delta_edofs.tail(np).setZero();

            result += delta_dJ_dxp.tail(np) 
                -  m_linesearch_object.apply_hessian_defoVarsIn_restVarsOut(delta_edofs).tail(np)
                -  d3E_y.tail(np);
            result *= coeff_J;
        }
    }

    return result;  
}


/*************************************************************
 ****************   UPDATE ADJOINT STATE   *******************
 *************************************************************/

template<template<typename> class Object_T, class EType>
bool EquilibriumOptimizationObject<Object_T,EType>::m_updateAdjointState(const Eigen::Ref<const Eigen::VectorXd> &params, EType opt_eType) {
    m_updateEquilibria(params);
    if (m_adjointStateIsCurrent.first && (m_adjointStateIsCurrent.second == opt_eType)) return false;

    // Solve the adjoint problems needed to efficiently evaluate the gradient.
    // Note: if the Hessian modification failed (tau runaway), the adjoint state
    // solves will fail. To keep the solver from giving up entirely, we simply
    // set the adjoint state to 0 in these cases. Presumably this only happens
    // at bad iterates that will be discarded anyway.
    try {
        // Adjoint solve for the target fitting objective on the deployed linkage
        this->objective.updateAdjointState(getEquilibriumOptimizer(), *m_linesearch_ws, opt_eType);
    }
    catch (...) {
        std::cout << "WARNING: Adjoint state solve failed" << std::endl;
        this->objective.clearAdjointState();
    }

    m_adjointStateIsCurrent = std::make_pair(true, opt_eType);


    return true;
}

/*************************************************************
 ********************   EQUILIBRIUM   ************************
 *************************************************************/

template<template<typename> class Object_T, class EType>
void EquilibriumOptimizationObject<Object_T,EType>::m_forceEquilibriumUpdate(){
    m_equilibrium_solve_successful = true;

    // Initialize the working set by copying the committed working
    // set if it exists; otherwise, construct a fresh one.
    if (m_committed_ws){
        // default copy
        m_linesearch_ws = std::make_unique<WorkingSet>(*m_committed_ws);
    }
    //if (m_committed_ws) m_linesearch_ws = std::make_unique<WorkingSet>(*m_committed_ws);
    else m_linesearch_ws = std::make_unique<WorkingSet>(getEquilibriumOptimizer().get_problem());


    try {
        if (m_equilibrium_options.verbose)
            std::cout << "Equilibrium solve" << std::endl;
        auto cr = getEquilibriumOptimizer().optimize(*m_linesearch_ws);
        // A backtracking failure will happen if the gradient tolerance is set too low
        // and generally does not indicate a complete failure/bad estimate of the equilibrium.
        // We therefore accept such equilibria with a warning.
        bool acceptable_failed_equilibrium = cr.backtracking_failure;
        if (!cr.success && !acceptable_failed_equilibrium) {
            throw std::runtime_error("Equilibrium solve did not converge");
        }
        if (acceptable_failed_equilibrium) {
            std::cout << "WARNING: equillibrium solve backtracking failure." << std::endl;
        }
    }
    catch (const std::runtime_error &e) {
        std::cout << "Equilibrium solve failed: " << e.what() << std::endl;
        m_equilibrium_solve_successful = false;
        return; // subsequent update_factorizations will fail if we caught a Tau runaway...
    }

    auto &uo = getEquilibriumOptimizer();
    // Factorize the Hessian for the updated linesearch equilibrium (not the second-to-last iteration).
    if (uo.solver().hasStashedFactorization() &&
            (m_linesearch_object.getRestVars() - m_object.getRestVars()).squaredNorm() == 0) {
        // We are re-evaluating at the committed equilibrium; roll back to the committed factorization
        // rather than re-computing it!
        uo.solver().swapStashedFactorization(); // restore the committed factorization
        uo.solver().      stashFactorization(); // stash a copy of it (TODO: avoid copy?)
    }
    else {
        // Use the final equilibrium's Hessian for sensitivity analysis, not the second-to-last iterates'
        try {
            uo.update_factorizations(*m_linesearch_ws);
            if (!uo.solver().hasStashedFactorization()) uo.solver().stashFactorization();
        }
        catch (const std::runtime_error &e) {
            std::cout << "Hessian factorization at equilibrium failed failed: " << e.what() << std::endl;
            m_equilibrium_solve_successful = false;
            return;
        }
    }

    // The cached adjoint state is invalidated whenever the equilibrium is updated...
    m_adjointStateIsCurrent.first = false;
    m_autodiffObjectIsCurrent   = false;

    this->objective.update();
}

template<template<typename> class Object_T, class EType>
bool EquilibriumOptimizationObject<Object_T,EType>::m_updateEquilibria(const Eigen::Ref<const Eigen::VectorXd> &newParams){
    BENCHMARK_SCOPED_TIMER_SECTION timer("m_updateEquilibria");
    if ( (size_t) newParams.size() != numParams()) throw std::runtime_error("Parameter vector size mismatch");

    if ((m_linesearch_object.getRestVars() - newParams).norm() < 1e-16) return false;
    m_linesearch_object.set(m_object);


    const Eigen::VectorXd currParams = m_object.getRestVars();
    Eigen::VectorXd delta_p = newParams - currParams;


    if (delta_p.squaredNorm() == 0) { // returning to linesearch start; no prediction/Hessian factorization necessary
        m_forceEquilibriumUpdate();
        return true;
    }

    // Apply the new design parameters and measure the energy with the 0^th order prediction
    // (i.e. at the currently committed equilibrium).
    // We will only replace this equilibrium if the higher-order predictions achieve a lower energy.
    m_linesearch_object.setRestVars(newParams);
    Real bestEnergy = m_linesearch_object.energy();
    Eigen::VectorXd curr_x = m_object.getDefoVars();
    Eigen::VectorXd best_x = curr_x;

    if (prediction_order > PredictionOrder::Zero) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("Predict equilibrium");
        // Return to using the Hessian for the last committed linkage
        // (i.e. for the equilibrium stored in m_flat and m_deployed).
        auto &opt_object = getEquilibriumOptimizer();
        if (!opt_object.solver().hasStashedFactorization())
            throw std::runtime_error("Factorization was not stashed... was commitLinesearchLinkage() called?");
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
            // m_delta_x = opt_object.solver().solve(neg_d2E_dxdp_deltap);
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

template<template<typename> class Object_T, class EType>
void EquilibriumOptimizationObject<Object_T,EType>::commitLinesearchObject() {
    m_object.set(m_linesearch_object);
    // Stash the current factorizations to be reused at each step of the linesearch
    // to predict the equilibrium at the new design parameters.
    getEquilibriumOptimizer().solver().stashFactorization();
    m_committed_ws = std::make_unique<WorkingSet>(*m_linesearch_ws);
}

template<template<typename> class Object_T, class EType>
void EquilibriumOptimizationObject<Object_T,EType>::invalidateEquilibria(){
    const Eigen::VectorXd  committedParams =            m_object.getRestVars();
    const Eigen::VectorXd linesearchParams = m_linesearch_object.getRestVars();

    getEquilibriumOptimizer().solver().clearStashedFactorization();
    m_linesearch_object.set(m_object);
    m_forceEquilibriumUpdate();           // recompute the equilibrium for the commited design parameters; done without any prediction
    commitLinesearchObject();           // commit the updated equilibrium, stashing the updated factorization
    m_updateEquilibria(linesearchParams); // recompute the linesearch equilibrium (if the linesearch params differ from the committed ones).
}

template<template<typename> class Object_T, class EType>
void EquilibriumOptimizationObject<Object_T,EType>::newPt(const Eigen::VectorXd &params){
    std::cout << "newPt at dist " << (m_object.getRestVars() - params).norm() << std::endl;
    m_updateAdjointState(params,EType::Full); // update the adjoint state as well as the equilibrium, since we'll probably be needing gradients at this point.
    commitLinesearchObject();
}