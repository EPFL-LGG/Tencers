
#ifndef TENCER_EQUILIBRIUM_PROBLEM_HH
#define TENCER_EQUILIBRIUM_PROBLEM_HH

#include <utility>
#include <algorithm>
#include <Eigen/Core>
#include "Tencer.hh"
#include <MeshFEM/newton_optimizer/newton_optimizer.hh>

#include <functional>

using CallbackFunction = std::function<void(NewtonProblem &, size_t)>;

template<typename Object>
struct TencerEquilibriumProblem : public NewtonProblem {

    TencerEquilibriumProblem(Object &obj): 
        object(obj), 
        m_hessianSparsity(obj.hessianSparsityPattern()), 
        m_characteristicLength(obj.characteristicLength()){
        }

    virtual void setVars(const VXd &vars) override {
        object.setVars(vars.cast<Real>());
    }
    virtual const VXd getVars() const override { return object.getVars().template cast<double>(); }
    virtual size_t numVars() const override { return object.numVars(); }

    virtual Real energy() const override {
        Real result = object.energy();
        if (result > elasticEnergyIncreaseFactorLimit * std::max(m_currElasticEnergy, energyLimitingThreshold))
             return safe_numeric_limits<Real>::max();
        return result;
    }

    virtual VXd gradient(bool freshIterate = false) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("EquilibriumProblem.gradient");
        auto result = object.gradient(freshIterate);
        return result.template cast<double>();
    }

    virtual SuiteSparseMatrix hessianSparsityPattern() const override { 
        if (!object.sparsityPatternUpdated){
            m_hessianSparsity = object.hessianSparsityPattern();
            object.sparsityPatternUpdated = true;
        }
        return m_hessianSparsity;
    }

    void setCustomIterationCallback(const CallbackFunction &cb) { m_customCallback = cb; }

    // "Physical" distance of a step relative to some characteristic lengthscale of the problem.
    // (Useful for determining reasonable step lengths to take when the Newton step is not possible.)
    // Note: overridden since the estimation of velocity only needs the DER dofs and not the material variables   
    virtual Real characteristicDistance(const Eigen::VectorXd &d) const override {
        return object.approxLinfVelocity(d.head(numVars())) / m_characteristicLength;
    }


    Real elasticEnergyIncreaseFactorLimit = safe_numeric_limits<Real>::max();
    Real energyLimitingThreshold = 1e-6;
    Real hessianShift = 0.0;  // Multiple of the identity to add to the Hessian on each evaluation.

    protected:
        
    virtual void m_evalHessian(SuiteSparseMatrix &result, bool projectionMask) const override {
        result.data().setZero();
        object.hessian(result, projectionMask);
        if (hessianShift != 0.0) result.addScaledIdentity(hessianShift);
    }

    virtual void m_evalMetric(SuiteSparseMatrix &result) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("EquilibriumSolver.m_evalMetric");
        result.setZero();
        object.massMatrix(result, true, true);
    }

    virtual void m_iterationCallback(size_t i) override {
        if (elasticEnergyIncreaseFactorLimit == safe_numeric_limits<Real>::max())
            m_currElasticEnergy = safe_numeric_limits<Real>::max();
        else m_currElasticEnergy = object.energy();
        object.updateParametrization();
        if (m_customCallback) return m_customCallback(*this, i);
    }

    Object &object;
    Real m_currElasticEnergy = safe_numeric_limits<Real>::max();
    CallbackFunction m_customCallback;
    mutable SuiteSparseMatrix m_hessianSparsity;
    Real m_characteristicLength = 1.0;
    

};

template<typename Object>
std::unique_ptr<NewtonOptimizer> get_equilibrium_optimizer(Object &obj,
                                                           const std::vector<size_t> &fixedVars,
                                                           const NewtonOptimizerOptions &opts, 
                                                           CallbackFunction customCallback,
                                                           Real systemEnergyIncreaseFactorLimit = safe_numeric_limits<Real>::max(), 
                                                           Real energyLimitingThreshold = 1e-6,
                                                           Real hessianShift = 0.0) {
    auto problem = std::make_unique<TencerEquilibriumProblem<Object>>(obj);
    problem->addFixedVariables(fixedVars);
    problem->setCustomIterationCallback(customCallback);
    problem->elasticEnergyIncreaseFactorLimit = systemEnergyIncreaseFactorLimit;
    problem->energyLimitingThreshold = energyLimitingThreshold;
    problem->hessianShift = hessianShift;
    auto opt = std::make_unique<NewtonOptimizer>(std::move(problem));
    opt->options = opts;
    return opt;
}

template<typename Object>
ConvergenceReport compute_equilibrium(Object &obj, 
                                      const std::vector<size_t> &fixedVars = std::vector<size_t>(),
                                      const NewtonOptimizerOptions &opts = NewtonOptimizerOptions(),
                                      CallbackFunction customCallback = nullptr,
                                      Real systemEnergyIncreaseFactorLimit = safe_numeric_limits<Real>::max(),
                                      Real energyLimitingThreshold = 1e-6,
                                      Real hessianShift = 0.0) {
    return get_equilibrium_optimizer(obj, fixedVars, opts, customCallback, systemEnergyIncreaseFactorLimit, energyLimitingThreshold, hessianShift)->optimize();
}


#endif