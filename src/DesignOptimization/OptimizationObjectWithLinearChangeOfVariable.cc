#include "OptimizationObjectWithLinearChangeOfVariable.hh"

template<template<typename> class Object_T, class EType>
OptimizationObjectWithLinearChangeOfVariable<Object_T,EType>::OptimizationObjectWithLinearChangeOfVariable(Object &object, CSCMat &matrix, std::shared_ptr<Base> opt_object) : 
    OptimizationObject<Object_T,EType>(object), 
    optimization_object(opt_object),
    transformation_matrix(matrix){}

template<template<typename> class Object_T, class EType>
Real OptimizationObjectWithLinearChangeOfVariable<Object_T,EType>::J(const Eigen::Ref<const Eigen::VectorXd> &params, EType opt_eType) {
    return optimization_object->J(apply_transformation(params), opt_eType); 
}

template<template<typename> class Object_T, class EType>
Eigen::VectorXd OptimizationObjectWithLinearChangeOfVariable<Object_T,EType>::gradp_J(const Eigen::Ref<const Eigen::VectorXd> &params, EType opt_eType){
    return apply_transformation_transpose(optimization_object->gradp_J(apply_transformation(params), opt_eType)); 
}

template<template<typename> class Object_T, class EType>
Eigen::VectorXd OptimizationObjectWithLinearChangeOfVariable<Object_T,EType>::apply_hess(const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, Real coeff_J, EType opt_eType){
    return apply_transformation_transpose(optimization_object->apply_hess(apply_transformation(params), apply_transformation(delta_p), coeff_J, opt_eType)); 
}

template<template<typename> class Object_T, class EType>
Eigen::VectorXd OptimizationObjectWithLinearChangeOfVariable<Object_T,EType>::apply_transformation(const Eigen::VectorXd &sv) const{
    return transformation_matrix.apply(sv);
}

template<template<typename> class Object_T, class EType>
Eigen::VectorXd OptimizationObjectWithLinearChangeOfVariable<Object_T,EType>::apply_transformation_transpose(const Eigen::VectorXd &ov) const{
    return transformation_matrix.apply(ov, true);
}