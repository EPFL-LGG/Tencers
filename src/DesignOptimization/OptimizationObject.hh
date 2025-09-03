#ifndef OPTIMIZATIONOBJECT_HH
#define OPTIMIZATIONOBJECT_HH

#include "design_optimization_terms.hh"

template<template<typename> class Object_T, class EType>
struct OptimizationObject {

    using Object = Object_T<Real>;

    DesignOptimizationObjective<Object_T, EType> objective;

    // Constructor
    OptimizationObject(Object &object) : m_object(object){}
    virtual ~OptimizationObject(){};

    // Objective function
    virtual Real J(const Eigen::Ref<const Eigen::VectorXd> &params, EType opt_eType = EType::Full) = 0;
    Real J(){ return J(params()); }

    // Gradient of the objective over the design parameters
    virtual Eigen::VectorXd gradp_J(const Eigen::Ref<const Eigen::VectorXd> &params, EType opt_eType = EType::Full) = 0;

    // Hessian matvec: H delta_p
    Eigen::VectorXd apply_hess_J(const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, EType opt_eType = EType::Full) { return apply_hess(params, delta_p, 1.0, opt_eType); }
    virtual Eigen::VectorXd apply_hess (const Eigen::Ref<const Eigen::VectorXd> &/*params*/, const Eigen::Ref<const Eigen::VectorXd> &/*delta_p*/, Real /*coeff_J*/, EType /*opt_eType*/ = EType::Full) {
        throw std::runtime_error("Method apply_hess has not been implemented");
    }

    // Design variables
    virtual size_t numParams() const = 0;
    virtual const Eigen::VectorXd params() const = 0;

    // Optimizer new point callback (to evaluate a new set of parameters)
    virtual void newPt(const Eigen::VectorXd &params){
        std::cout << "newPt at dist " << (m_object.getRestVars() - params).norm() << std::endl;
    }

protected: 
    Object &m_object;
};





#endif /* end of include guard: OPTIMIZATIONOBJECT_HH */