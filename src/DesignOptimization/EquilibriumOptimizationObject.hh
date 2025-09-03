#ifndef EQUILIBRIUMOPTIMIZATIONOBJECT_HH
#define EQUILIBRIUMOPTIMIZATIONOBJECT_HH

#include <MeshFEM/Geometry.hh>
#include "design_optimization_terms.hh"
#include <MeshFEM/newton_optimizer/newton_optimizer.hh>
#include <MeshFEM/Types.hh>
#include "OptimizationObject.hh"
#include "../Tencer.hh"

enum class OptAlgorithm    : int { NEWTON_CG=0, BFGS=1 };
enum class PredictionOrder : int { Zero = 0, One = 1, Two = 2};

template<template<typename> class Object_T, class EType>
struct EquilibriumOptimizationObject : public OptimizationObject<Object_T, EType> {

    using   Object = Object_T<Real>;
    using ADObject = Object_T<ADReal>;
    using Base = OptimizationObject<Object_T,EType>;
    using Base::m_object;

    PredictionOrder prediction_order = PredictionOrder::One;

    // Constructor
    EquilibriumOptimizationObject(Object &object, const NewtonOptimizerOptions &eopts, const std::vector<size_t> &fixedVars,Real hessian_shift, PredictionOrder po);
    virtual ~EquilibriumOptimizationObject(){};

    // Objective function
    virtual Real J(const Eigen::Ref<const Eigen::VectorXd> &params, EType opt_eType = EType::Full) override;

    // Gradient of the objective over the design parameters
    virtual Eigen::VectorXd gradp_J(const Eigen::Ref<const Eigen::VectorXd> &params, EType opt_eType = EType::Full) override;

    // Hessian matvec: H delta_p
    virtual Eigen::VectorXd apply_hess  (const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, Real coeff_J, EType opt_eType = EType::Full) override;

    // Design variables
    virtual size_t numParams() const override {return m_numParams;}
    virtual const Eigen::VectorXd params() const override {return m_object.getRestVars();}

    // Equilibrium Helpers.
    void commitLinesearchObject();
    NewtonOptimizer &getEquilibriumOptimizer() {return *m_equilibrium_optimizer;}
    NewtonOptimizerOptions getEquilibriumOptions() const {return m_equilibrium_options;};

    // When the fitting weights change the adjoint state must be recomputed.
    // Let the user manually inform us of this change.
    void invalidateAdjointState() {m_adjointStateIsCurrent.first = false;}

    // To be called after modifying the simulation energy formulation.
    // In this case, the committed and linesearch equilibra, as well as any
    // cached Hessian factorizations and adjoint states, are invalid and must
    // be recomputed.
    void invalidateEquilibria(); 

    const Object &    linesearchObject() const {return m_linesearch_object;}
    const Object &     committedObject() const {return m_object;}
    const WorkingSet   &linesearchWorkingSet() const { 
        if (!m_linesearch_ws) throw std::runtime_error("Unallocated working set"); 
        return *m_linesearch_ws; 
    }
    const WorkingSet   & committedWorkingSet() const { 
        if (!m_committed_ws) throw std::runtime_error("Unallocated working set"); 
        return  *m_committed_ws;
    }

    // Evaluate at a new set of parameters and commit this change (which
    // are used as a starting point for solving the line search equilibrium)
    virtual void newPt(const Eigen::VectorXd &params) override;

protected: 

    Object m_linesearch_object;
    ADObject m_diff_object;

    size_t m_numParams;
    std::vector<size_t> m_fixedVars;

    // equilibrium solve
    NewtonOptimizerOptions m_equilibrium_options;
    bool m_equilibrium_solve_successful;

    std::unique_ptr<NewtonOptimizer> m_equilibrium_optimizer;
    std::unique_ptr<WorkingSet> m_linesearch_ws, m_committed_ws; // Working set at the linesearch/committed object

    bool m_autodiffObjectIsCurrent = false;
    std::pair<bool, EType> m_adjointStateIsCurrent;

    Eigen::VectorXd m_delta_y, m_delta_x;

    // update adjoint state
    virtual bool m_updateAdjointState(const Eigen::Ref<const Eigen::VectorXd> &params, EType opt_eType);

    // Forward equilibrium 
    virtual void m_forceEquilibriumUpdate();

    virtual bool m_updateEquilibria(const Eigen::Ref<const Eigen::VectorXd> &newParams);

};





#endif /* end of include guard: EQUILIBRIUMOPTIMIZATIONOBJECT_HH */