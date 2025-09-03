////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Terms for the design optimization, where the objective and constraints are
//  expressed in terms of the equilbrium x^*(p) (which is considered to be a
//  function of the design parameters p).
//
//  Objective terms are of the form w_i * J_i(x, p), where w_i is term's weights
//  Constraint terms are of the form c_i(x, p).
//
//  Then the full objective is the the form J(p) = sum_i w_i J_i(x^*(p)), p.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  04/26/2020 17:57:37
////////////////////////////////////////////////////////////////////////////////
#ifndef DESIGNOPTIMIZATIONTERMS_HH
#define DESIGNOPTIMIZATIONTERMS_HH

#include <memory>
#include <algorithm>

#include <MeshFEM/AutomaticDifferentiation.hh>
#include <MeshFEM/newton_optimizer/newton_optimizer.hh>
#include "../Tencer.hh"
// #include <MeshFEM/ElasticObject.hh>

template<template<typename> class Object_T>
struct DesignOptimizationObjectTraits {
    template<typename T> static size_t     numDefoVars      (const Object_T<T> &obj) { return obj.numDefoVars(); }
    template<typename T> static size_t     numRestVars   (const Object_T<T> &obj) { return obj.numRestVars(); }
    template<typename T> static size_t     numAugmentedVars(const Object_T<T> &obj) { return numDefoVars(obj) + numRestVars(obj); }
    template<typename T> static auto       getAugmentedVars(const Object_T<T> &obj) { return obj.getVars(VariableMask::All); }
    template<typename T> static void       setAugmentedVars(      Object_T<T> &obj, const VecX_T<T> &v) { obj.setVars(v, VariableMask::All); }
};

template<template<typename> class Object_T>
struct DesignOptimizationTerm {
    using   Object = Object_T<Real>;
    using ADObject = Object_T<ADReal>;
    using  OTraits = DesignOptimizationObjectTraits<Object_T>;
    using      VXd = Eigen::VectorXd;

    DesignOptimizationTerm(const Object &obj) : m_obj(obj) { }

    size_t numVars()       const { return OTraits::numAugmentedVars(m_obj); }
    size_t numDefoVars()    const { return OTraits::numDefoVars(m_obj); }
    size_t numRestVars() const { return OTraits::numRestVars(m_obj); }

    Real unweightedValue() const { return m_value(); }
    Real value() const { return m_weight() * m_value(); }

    // partial value / partial xp
    VXd grad() const { return m_weight() * cachedGrad(); }

    // partial value / partial x
    VXd grad_x() const { return m_weight() * cachedGrad().head(numDefoVars()); }

    // partial value / partial p
    VXd grad_p() const { return m_weight() * cachedGrad().tail(numRestVars()); }

    // (d^2 J / d{x, p} d{x, p}) [ delta_x, delta_p ]
    // Note: autodiffObject should have [delta_x, delta_p] already injected.
    VXd delta_grad(Eigen::Ref<const VXd> delta_xp, const ADObject &autodiffObject) const { return m_weight() * m_delta_grad(delta_xp, autodiffObject); }

    // Reset the cache of the partial derivatives wrt x and p
    void update() { m_update(); m_cachedGrad.resize(0); }

    const VXd cachedGrad() const; 

    virtual ~DesignOptimizationTerm() { }

    ////////////////////////////////////////////////////////////////////////////
    // For validation/debugging
    ////////////////////////////////////////////////////////////////////////////
    const Object &object() const { return m_obj; }
    Real computeValue() const {return m_weight() * m_value();}
    VXd computeGrad() const { return m_weight() * m_grad(); }
    VXd computeDeltaGrad(Eigen::Ref<const VXd> delta_xp) const; 
    Real getWeight() const { return m_weight(); }
    virtual void setWeight(Real) { throw std::runtime_error("Weight cannot be changed."); }

protected:
    const Object &m_obj;
    mutable VXd m_cachedGrad;

    virtual Real m_weight() const = 0; // Always 1.0 for constraints, custom weight for objective terms.

    // Methods that must be implemented by each term...
    virtual Real  m_value() const = 0;

    // [partial value/partial x, partial value/ partial p]
    virtual VXd  m_grad() const = 0;
    // delta [partial value/partial x, partial value/ partial p]
    virtual VXd m_delta_grad(Eigen::Ref<const VXd> delta_xp, const ADObject &autodiffObject) const = 0;

    // (Optional) update cached quantities shared between energy/gradient calculations.
    virtual void m_update() { }
};

template<template<typename> class Object_T>
struct DesignOptimizationObjectiveTerm : public DesignOptimizationTerm<Object_T> {
    using DOT  = DesignOptimizationTerm<Object_T>;
    using DOT::DOT;
    Real weight = 1.0;

    virtual void setWeight(Real w) override { weight = w; }
protected:
    virtual Real m_weight() const override { return weight; }
};


template<template<typename> class Object_T, class OptimizationTerm, typename EType>
struct AdjointState {
    using Derived = OptimizationTerm;

    using DOT = DesignOptimizationTerm<Object_T>;
    using VXd = Eigen::VectorXd;
    using   Object = typename DOT::  Object;
    using ADObject = typename DOT::ADObject;

    const OptimizationTerm &derived() const { return *static_cast<const OptimizationTerm *>(this); }
          OptimizationTerm &derived()       { return *static_cast<      OptimizationTerm *>(this); }

    // Adjoint solve for the optimization term:
    //      [H_3D a][w_x     ] = [dJ/dx]   or    H_3D w_x = dJ/dx
    //      [a^T  0][w_lambda]   [  0  ]
    // depending on whether average angle actuation is applied.
    // Note : before, we got a reduced problem by removing fixed variables from the problem, we solved it, and then we got back to the original problem by setting the fixed variables to 0
    // Now this is taken care of in the optimizer
    void updateAdjointState(NewtonOptimizer &opt, EType etype = EType::Full); 

    // Solve for the adjoint state in the case that bound constraints are active.
    // This should have 0s in the components corresponding to the constrained variables.
    void updateAdjointState(NewtonOptimizer &opt, const WorkingSet &ws, EType etype = EType::Full) {
        updateAdjointState(opt, etype);
        ws.getFreeComponent(m_adjointState);
    }

    const VXd &adjointState() const { return m_adjointState; }
    void clearAdjointState() { m_adjointState.setZero(); }

protected:
    VXd m_adjointState, m_deltaAdjointState;
};


// Collect objective terms depending on a single object/equilibrium.
// (All of these collectively will be given a single adjoint state).
template<template<typename> class Object_T, class EType>
struct DesignOptimizationObjective : public AdjointState<Object_T, DesignOptimizationObjective<Object_T, EType>, EType> {
    using DOT      = DesignOptimizationTerm<Object_T>;
    using ADObject = typename DOT::ADObject;
    using VXd      = Eigen::VectorXd;

    using TermPtr = std::shared_ptr<DOT>;
    struct TermRecord {
        TermRecord(const std::string &n, EType t, TermPtr tp) : name(n), type(t), term(tp) { }
        std::string name;
        EType type;
        TermPtr term;
    };

    void update() {
        for (TermRecord &t : terms)
            t.term->update();
    }

    Real value(EType type = EType::Full) const;

    Real operator()(EType type = EType::Full) const { return value(type); }

    // Note: individual terms return their *unweighted* values.
    // Real value(const std::string &name) const;

    VXd grad(EType type = EType::Full) const;

    VXd grad_x(EType type = EType::Full) const {
        return grad(type).head(terms.at(0).term->numDefoVars());
    }

    VXd grad_p(EType type = EType::Full) const {
        return grad(type).tail(terms.at(0).term->numRestVars());
    }

    VXd computeGrad(EType type = EType::Full) const;

    // (d^2 J / d{x, p} d{x, p}) [ delta_x, delta_p ]
    // Note: autodiffObject should have [delta_x, delta_p] already injected.
    VXd delta_grad(Eigen::Ref<const VXd> delta_xp, const ADObject &autodiffObject, EType type = EType::Full) const;

    VXd computeDeltaGrad(Eigen::Ref<const VXd> delta_xp, EType type = EType::Full) const;

    std::map<std::string, Real> values() const;

    std::map<std::string, Real> weightedValues() const;

    void printReport() const;

    void add(const std::string &name, EType etype, TermPtr term) {
        if (m_find(name) != terms.end()) throw std::runtime_error("Term " + name + " already exists");
        terms.emplace_back(name, etype, term);
    }

    void add(const std::string &name, EType etype, TermPtr term, Real weight) {
        add(name, etype, term);
        terms.back().term->setWeight(weight);
    }

    void remove(const std::string &name) {
        auto it = m_find(name);
        if (it == terms.end()) throw std::runtime_error("Term " + name + " doesn't exist");
        terms.erase(it);
    }

    DOT &get(const std::string &name) {
        auto it = m_find(name);
        if (it == terms.end()) throw std::runtime_error("Term " + name + " doesn't exist");
        return *(it->term);
    }

    const DOT &get(const std::string &name) const {
        auto it = m_find(name);
        if (it == terms.end()) throw std::runtime_error("Term " + name + " doesn't exist");
        return *(it->term);
    }

    std::vector<TermRecord> terms;

private:
    decltype(terms.cbegin()) m_find(const std::string &name) const {
        return std::find_if(terms.begin(), terms.end(),
                            [&](const TermRecord &t) { return t.name == name; });
    }
};

#endif /* end of include guard: DESIGNOPTIMIZATIONTERMS_HH */