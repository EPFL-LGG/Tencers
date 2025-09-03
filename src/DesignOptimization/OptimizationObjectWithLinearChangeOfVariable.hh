#ifndef OPTIMIZATIONOBJECTWITHLINEARCHANGEOFVARIABLE_HH
#define OPTIMIZATIONOBJECTWITHLINEARCHANGEOFVARIABLE_HH

#include "OptimizationObject.hh"

template<template<typename> class Object_T, class EType>
struct OptimizationObjectWithLinearChangeOfVariable : public OptimizationObject<Object_T,EType>{

    using   Object = Object_T<Real>;
    using ADObject = Object_T<ADReal>;
    using CSCMat = CSCMatrix<SuiteSparse_long, Real>;
    using Base = OptimizationObject<Object_T,EType>;
    using Base::m_object;

    // Tracks an OptimizationObject
    std::shared_ptr<Base> optimization_object;

    // Transformation matrix
    CSCMat transformation_matrix;

    // Constructor
    OptimizationObjectWithLinearChangeOfVariable(Object &object, CSCMat &matrix, std::shared_ptr<Base> opt_object);

    // Objective function
    virtual Real J(const Eigen::Ref<const Eigen::VectorXd> &params, EType opt_eType = EType::Full) override;

    // Gradient of the objective over the design parameters
    virtual Eigen::VectorXd gradp_J(const Eigen::Ref<const Eigen::VectorXd> &params, EType opt_eType = EType::Full) override;

    // Hessian matvec: H delta_p
    virtual Eigen::VectorXd apply_hess(const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, Real coeff_J, EType opt_eType = EType::Full) override;

    // Design variables
    virtual size_t numParams() const override {return transformation_matrix.n;}
    // virtual const Eigen::VectorXd params() const = 0; // This one needs to be defined by the user

    // Optimizer new point callback (to evaluate a new set of parameters)
    virtual void newPt(const Eigen::VectorXd &params) override {optimization_object->newPt(apply_transformation(params));}

    // From symmetric variables to original variables
    Eigen::VectorXd apply_transformation(const Eigen::VectorXd &sv) const;

    // From original variables to symmetric variables
    Eigen::VectorXd apply_transformation_transpose(const Eigen::VectorXd &ov) const;

};
#endif