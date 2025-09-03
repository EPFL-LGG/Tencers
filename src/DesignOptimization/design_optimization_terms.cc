#include "design_optimization_terms.hh"

template<template<typename> class Object_T>
const Eigen::VectorXd DesignOptimizationTerm<Object_T>::cachedGrad() const {
    if (size_t(m_cachedGrad.rows()) != numVars())
        m_cachedGrad = m_grad();
    return m_cachedGrad;
}


template<template<typename> class Object_T>
Eigen::VectorXd DesignOptimizationTerm<Object_T>::computeDeltaGrad(Eigen::Ref<const VXd> delta_xp) const {
    if (size_t(delta_xp.size()) != numVars()) throw std::runtime_error("Size mismatch");
    ADObject diff_obj(m_obj);
    VecX_T<ADReal> ad_dofs = OTraits::getAugmentedVars(m_obj);
    const size_t nv = numVars();
    for (size_t i = 0; i < nv; ++i) ad_dofs[i].derivatives()[0] = delta_xp[i];
    OTraits::setAugmentedVars(diff_obj, ad_dofs);
    return m_weight() * m_delta_grad(delta_xp, diff_obj);
}

template<template<typename> class Object_T, class OptimizationTerm, typename EType>
void AdjointState<Object_T, OptimizationTerm, EType>::updateAdjointState(NewtonOptimizer &opt, EType etype) {
    m_adjointState.resize(opt.get_problem().numVars()); // ensure correct size in case adjoint solve fails...
    if (opt.get_problem().hasLEQConstraint()) {
        m_adjointState = opt.kkt_solver(opt.solver(), derived().grad_x(etype));
    }
    else {
        m_adjointState = opt.extractFullSolution(opt.solver().solve(opt.removeFixedEntries(derived().grad_x(etype))));
    }
}

template<template<typename> class Object_T, class EType>
Real DesignOptimizationObjective<Object_T,EType>::value(EType type) const {
    Real val = 0.0;
    for (const TermRecord &t : terms) {
        if (t.term->getWeight() == 0.0) continue;
        if ((type == EType::Full) || (type == t.type))
            val += t.term->value();
    }
    return val;
}



template<template<typename> class Object_T, class EType>
Eigen::VectorXd DesignOptimizationObjective<Object_T,EType>::grad(EType type) const {
    if (terms.empty()) throw std::runtime_error("no terms present");
    VXd result = VXd::Zero(terms[0].term->numVars());
    for (const TermRecord &t : terms) {
        if (t.term->getWeight() == 0.0) continue;
        if ((type == EType::Full) || (type == t.type))
            result += t.term->grad();
    }
    return result;
}

template<template<typename> class Object_T, class EType>
Eigen::VectorXd DesignOptimizationObjective<Object_T,EType>::computeGrad(EType type) const {
    if (terms.empty()) throw std::runtime_error("no terms present");
    VXd result = VXd::Zero(terms[0].term->numVars());
    for (const TermRecord &t : terms) {
        if ((type == EType::Full) || (type == t.type))
            result += t.term->computeGrad();
    }
    return result;
}

template<template<typename> class Object_T, class EType>
Eigen::VectorXd DesignOptimizationObjective<Object_T,EType>::delta_grad(Eigen::Ref<const VXd> delta_xp, const ADObject &autodiffObject, EType type) const {
    if (terms.empty()) throw std::runtime_error("no terms present");
    VXd result = VXd::Zero(terms.front().term->numVars());
    for (const TermRecord &t : terms) {
        if (t.term->getWeight() == 0.0) continue;
        if ((type == EType::Full) || (type == t.type))
            result += t.term->delta_grad(delta_xp, autodiffObject);
    }
    return result;
}

template<template<typename> class Object_T, class EType>
Eigen::VectorXd DesignOptimizationObjective<Object_T,EType>::computeDeltaGrad(Eigen::Ref<const VXd> delta_xp, EType type) const {
    if (terms.empty()) throw std::runtime_error("no terms present");
    VXd result = VXd::Zero(terms.front().term->numVars());
    for (const TermRecord &t : terms) {
        if ((type == EType::Full) || (type == t.type))
            result += t.term->computeDeltaGrad(delta_xp);
    }
    return result;
}

template<template<typename> class Object_T, class EType>
std::map<std::string, Real> DesignOptimizationObjective<Object_T,EType>::values() const {
    std::map<std::string, Real> result;
    for (const TermRecord &t : terms)
        result[t.name] = t.term->unweightedValue();
    return result;
}

template<template<typename> class Object_T, class EType>
std::map<std::string, Real> DesignOptimizationObjective<Object_T,EType>::weightedValues() const {
    std::map<std::string, Real> result;
    for (const TermRecord &t : terms)
        result[t.name] = t.term->value();
    return result;
}

template<template<typename> class Object_T, class EType>
void DesignOptimizationObjective<Object_T,EType>::printReport() const {
    std::cout << "value " << value();
    for (const TermRecord &t : terms)
        std::cout << " " << t.name << t.term->unweightedValue();
    std::cout << std::endl;
}