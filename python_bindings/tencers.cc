#include <MeshFEM/FieldSamplerMatrix.hh>

#include <MeshFEM/Utilities/NameMangling.hh>
#include <MeshFEM/../../python_bindings/BindingInstantiations.hh>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>


#include <../src/DesignOptimization/design_optimization_terms.hh>
#include <../src/DesignOptimization/design_optimization_terms.cc>
#include <../src/DesignOptimization/OptimizationObject.hh>
#include <../src/DesignOptimization/EquilibriumOptimizationObject.hh>
#include <../src/DesignOptimization/EquilibriumOptimizationObject.cc>
#include <../src/DesignOptimization/OptimizationObjectWithLinearChangeOfVariable.hh>
#include <../src/DesignOptimization/OptimizationObjectWithLinearChangeOfVariable.cc>

#include "../src/Spring.hh"
#include "../src/Spring.cc"
#include "../src/helpers.hh"
#include "../src/Tencer.hh"
#include "../src/Tencer.cc"
#include "../src/tencers_knitro_problem.hh"
#include "../src/weighted_target_curve_fitter.hh"
#include "../src/tencer_optimization.hh"
#include "../src/tencer_optimization.cc"
#include "../src/symmetry_optimization.hh"
#include "../src/symmetry_optimization.cc"
#include "../src/SlidingNode.hh"
#include "../src/SlidingNode.cc"
#include "../src/TencerEquilibriumProblem.hh"


namespace py = pybind11;

// Hack around a limitation of pybind11 where we cannot specify argument passing policies and
// pybind11 tries to make a copy if the passed instance is not already registered:
//      https://github.com/pybind/pybind11/issues/1200
// We therefore make our Python callback interface use a raw pointer to forbid this copy (which
// causes an error since NewtonProblem is not copyable).
using PyCallbackFunction = std::function<bool(NewtonProblem *, size_t)>;
CallbackFunction callbackWrapper(const PyCallbackFunction &pcb) {
    return [pcb](NewtonProblem &p, size_t i) -> bool { if (pcb) return pcb(&p, i); return false; };
}

PYBIND11_MODULE(tencers, m) {
    m.doc() = "tencers codebase";

    py::module::import("MeshFEM");
    py::module::import("ElasticRods");

    py::module detail_module = m.def_submodule("detail");
    using OO = OptimizationObject<Tencer_T,OptEnergyType>;
    using EOO = EquilibriumOptimizationObject<Tencer_T,OptEnergyType>;
    using LOO = OptimizationObjectWithLinearChangeOfVariable<Tencer_T,OptEnergyType>;
    using Tencer = Tencer_T<Real>;
    using Spring = Spring_T<Real>;
    using RestVarsType = typename Spring::RestVarsType;
    using SpringAnchorVars = SpringAnchorVars_T<Real>;
    using Node = Node_T<Real>;
    using FixedNode = FixedNode_T<Real>;
    using SlidingNode = SlidingNode_T<Real>;
    using LinearSlidingNode = LinearSlidingNode_T<Real>;
    using LinearClosedSlidingNode = LinearClosedSlidingNode_T<Real>;
    using LinearOpenSlidingNode = LinearOpenSlidingNode_T<Real>;
    using KnotEnergyType = typename Tencer::KnotEnergyType;
    using EnergyType = typename Tencer::EnergyType;
    using Vec3 = Eigen::Matrix<double, 3, 1>;
    

    py::enum_<OptAlgorithm>(m, "OptAlgorithm")
        .value("NEWTON_CG", OptAlgorithm::NEWTON_CG)
        .value("BFGS",      OptAlgorithm::BFGS     )
        ;

    py::enum_<OptEnergyType>(m, "OptEnergyType")
        .value("Full", OptEnergyType::Full)
        .value("CurveFitting", OptEnergyType::CurveFitting )
        ;

    py::enum_<PredictionOrder>(m, "PredictionOrder")
        .value("Zero", PredictionOrder::Zero)
        .value("One", PredictionOrder::One)
        .value("Two", PredictionOrder::Two)
    ;

    py::enum_<KnotEnergyType>(m, "KnotEnergyType")
        .value("Full", KnotEnergyType::Full)
        .value("Elastic", KnotEnergyType::Elastic)
        .value("Springs", KnotEnergyType::Springs)
    ;

    py::enum_<RestVarsType>(m, "RestVarsType")
        .value("StiffnessOnVerticesFast",   RestVarsType::StiffnessOnVerticesFast)
        .value("Stiffness",                 RestVarsType::Stiffness)
        .value("SpringAnchors",             RestVarsType::SpringAnchors)
        .value("StiffnessAndSpringAnchors", RestVarsType::StiffnessAndSpringAnchors)
    ;

    py::enum_<VariableMask>(m, "VariableMask")
        .value("Defo", VariableMask::Defo)
        .value("Rest", VariableMask::Rest)
        .value( "All", VariableMask::All)
        ;

    using PySp = py::class_<Spring, std::shared_ptr<Spring>>;
    auto spring = PySp(m,"Spring");

    spring
        .def(py::init<Vec3 &, Vec3 &, double, double>(), py::arg("coordA"), py::arg("coordB"),py::arg("stiffness"),py::arg("rest_length"))
        .def(py::init<const Spring &>())  // copy constructor
        .def("get_coords", &Spring::get_coords)
        .def("energy",     &Spring::energy)
        .def("dE_dx",      py::overload_cast<size_t        >(&Spring::dE_dx,    py::const_), py::arg("i"))
        .def("dE_du",      py::overload_cast<size_t        >(&Spring::dE_du,    py::const_), py::arg("i"))
        .def("d2E_du2",    py::overload_cast<size_t, size_t>(&Spring::d2E_du2,  py::const_), py::arg("i"), py::arg("j"))
        .def("d2E_dx2",    py::overload_cast<size_t, size_t>(&Spring::d2E_dx2,  py::const_), py::arg("i"), py::arg("j"))
        .def("d2E_dxdu",   py::overload_cast<size_t, size_t>(&Spring::d2E_dxdu, py::const_), py::arg("xi"), py::arg("ui"))
        .def("d2E_dxdk",   py::overload_cast<size_t        >(&Spring::d2E_dxdk, py::const_), py::arg("i"))
        .def("dE_dx",      py::overload_cast<              >(&Spring::dE_dx,    py::const_))
        .def("dE_du",      py::overload_cast<              >(&Spring::dE_du,    py::const_))
        .def("dE_dk",      &Spring::dE_dk)
        .def("d2E_dx2",    py::overload_cast<              >(&Spring::d2E_dx2,  py::const_))
        .def("d2E_du2",    py::overload_cast<              >(&Spring::d2E_du2,  py::const_))
        .def("d2E_dxdu",   py::overload_cast<              >(&Spring::d2E_dxdu, py::const_))
        .def("d2E_dxdk",   py::overload_cast<              >(&Spring::d2E_dxdk, py::const_))
        .def_readonly("position_A", &Spring::position_A, py::return_value_policy::reference)
        .def_readonly("position_B", &Spring::position_B, py::return_value_policy::reference)
        .def_readwrite("stiffness", &Spring::stiffness)
        .def("get_rest_vars_type",  &Spring::get_rest_vars_type)
        .def("get_rest_length", &Spring::get_rest_length)

        // pickling
        // TODO
        ;

    py::class_<SpringAttachmentVertices, std::shared_ptr<SpringAttachmentVertices>>(m, "SpringAttachmentVertices")
        .def(py::init<int, int, int,int>(), py::arg("rodA"),py::arg("vertexA"),py::arg("rodB"),py::arg("vertexB"))
        .def_readonly("rodIdxA", &SpringAttachmentVertices::rod_idx_A, py::return_value_policy::reference)
        .def_readonly("rodIdxB", &SpringAttachmentVertices::rod_idx_B, py::return_value_policy::reference)
        .def_readonly("vertexA", &SpringAttachmentVertices::vertex_A, py::return_value_policy::reference)
        .def_readonly("vertexB", &SpringAttachmentVertices::vertex_B, py::return_value_policy::reference)

        // Comparison operator
        .def("__eq__", [](const SpringAttachmentVertices &sav, const SpringAttachmentVertices &other) {
            return sav == other;
        })

        // Pickling
        .def(py::pickle(
            [](const SpringAttachmentVertices &sav) {
                return py::make_tuple(sav.rod_idx_A, sav.vertex_A, sav.rod_idx_B, sav.vertex_B);
            },
            [](const py::tuple &state) {
                if (state.size() != 4)
                    throw std::runtime_error("Invalid state!");
                return SpringAttachmentVertices(
                    state[0].cast<int>(),
                    state[1].cast<int>(),
                    state[2].cast<int>(),
                    state[3].cast<int>()
                );
            }
        ))
        ;

    // ------------------------------------------------------------------------------------------
    //                                    Sliding Nodes
    // ------------------------------------------------------------------------------------------

    py::class_<SpringAnchorVars, std::shared_ptr<SpringAnchorVars>>(m, "SpringAnchorVars")
        .def(py::init<int, Real, int, Real>(), py::arg("rod_idx_A"), py::arg("mat_coord_A"), py::arg("rod_idx_B"), py::arg("mat_coord_B"))
        .def_readonly("rod_idx_A", &SpringAnchorVars::rod_idx_A, py::return_value_policy::reference)
        .def_readonly("rod_idx_B", &SpringAnchorVars::rod_idx_B, py::return_value_policy::reference)
        .def_readonly("mat_coord_A", &SpringAnchorVars::mat_coord_A, py::return_value_policy::reference)
        .def_readonly("mat_coord_B", &SpringAnchorVars::mat_coord_B, py::return_value_policy::reference)
        .def("__getitem__", [](SpringAnchorVars &sav, size_t i) -> Real { return sav[i]; })
        .def("getRodIdx", &SpringAnchorVars::getRodIdx, py::arg("i"))
        .def("setRodIdx", &SpringAnchorVars::setRodIdx, py::arg("i"), py::arg("ri"))

        // Comparison operator
        .def("__eq__", [](const SpringAnchorVars &sav, const SpringAnchorVars &other) {
            return sav == other;
        })
        
        // Pickling
        .def(py::pickle(
            [](const SpringAnchorVars &sav) {
                return py::make_tuple(sav.rod_idx_A, sav.mat_coord_A, sav.rod_idx_B, sav.mat_coord_B);
            },
            [](const py::tuple &state) {
                if (state.size() != 4)
                    throw std::runtime_error("Invalid state!");
                return SpringAnchorVars(
                    state[0].cast<int>(),
                    state[1].cast<Real>(),
                    state[2].cast<int>(),
                    state[3].cast<Real>()
                );
            }
        ))
    ;

    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
        .def("p", &Node::p)
    ;

    py::class_<FixedNode, Node, std::shared_ptr<FixedNode>>(m, "FixedNode");

    py::class_<SlidingNode, Node, std::shared_ptr<SlidingNode>>(m, "SlidingNode");

    py::class_<LinearSlidingNode, SlidingNode, std::shared_ptr<LinearSlidingNode>>(m, "LinearSlidingNode");

    py::class_<LinearClosedSlidingNode, LinearSlidingNode, std::shared_ptr<LinearClosedSlidingNode>>(m, "LinearClosedSlidingNode")
        .def("set",   &LinearClosedSlidingNode::set, py::arg("u"))
        .def("u",     &LinearClosedSlidingNode::u)
        .def("p",     &LinearClosedSlidingNode::p)
        .def("dp_du", &LinearClosedSlidingNode::dp_du)
        .def("dp_dx", &LinearClosedSlidingNode::dp_dx, py::arg("i"))
        .def("stencilIndex",     &LinearClosedSlidingNode::stencilIndex)
        .def("stencilIndices",   &LinearClosedSlidingNode::stencilIndices)
        .def("closestNodeIndex", &LinearClosedSlidingNode::closestNodeIndex)
        .def("x",    &LinearClosedSlidingNode::x,    py::arg("i"))
    ;

    py::class_<LinearOpenSlidingNode, LinearSlidingNode, std::shared_ptr<LinearOpenSlidingNode>>(m, "LinearOpenSlidingNode")
        .def("set",   &LinearOpenSlidingNode::set, py::arg("u"))
        .def("u",     &LinearOpenSlidingNode::u)
        .def("p",     &LinearOpenSlidingNode::p)
        .def("dp_du", &LinearOpenSlidingNode::dp_du)
        .def("dp_dx", &LinearOpenSlidingNode::dp_dx, py::arg("i"))
        .def("stencilIndex",     &LinearOpenSlidingNode::stencilIndex)
        .def("stencilIndices",   &LinearOpenSlidingNode::stencilIndices)
        .def("closestNodeIndex", &LinearOpenSlidingNode::closestNodeIndex)
        .def("x",    &LinearOpenSlidingNode::x,    py::arg("i"))
    ;

    using PyMTK = py::class_<Tencer, std::shared_ptr<Tencer>>;
    auto tencer = PyMTK(m, "Tencer");


    tencer
        // constructors
        .def(py::init<const std::vector<ElasticRod> &, const std::vector<PeriodicRod> &, const std::vector<Spring> &, std::vector<SpringAttachmentVertices> &, std::vector<ElasticRod> &>(), py::arg("open_rods"), py::arg("closed_rods"),py::arg("springs"),py::arg("attachment_vertices"), py::arg("targets"))
        .def(py::init<const Tencer &>(), py::arg("knot"))

        // rod and spring accessors
        .def("numRods", &Tencer::num_rods)
        .def("getOpenRods", &Tencer::get_open_rods)
        .def("getClosedRods", &Tencer::get_closed_rods)
        .def("getSprings", &Tencer::get_springs)
        .def("getAttachmentVertices", &Tencer::get_attachment_vertices)
        .def("getTargetRods", &Tencer::get_target_rods)

        // variable accessors
        .def("numDefoVars", &Tencer::numDefoVars)
        .def("numRestVars", &Tencer::numRestVars)
        .def("numVars", &Tencer::numVars, py::arg("vmask") = VariableMask::Defo)
        .def("getVars", &Tencer::getVars, py::arg("vmask") = VariableMask::Defo)
        .def("getDefoVars", &Tencer::getDefoVars)
        .def("getRestVars", &Tencer::getRestVars)
        .def("setVars", &Tencer::setVars, py::arg("new_vars"), py::arg("vmask") = VariableMask::Defo)
        .def("setDefoVars", &Tencer::setDefoVars, py::arg("new_defo_vars"))
        .def("setRestVars", &Tencer::setRestVars, py::arg("new_rest_vars"))
        .def("getDefoVarsIndexForRod", &Tencer::get_defo_var_index_for_rod, py::arg("rodIndex"))

        .def("getSpringAnchorVars", &Tencer::get_spring_anchor_vars)
        .def("setSpringAnchorVars", &Tencer::set_spring_anchor_vars, py::arg("sav"))
        .def("getSpringAnchorVarsForSpring", &Tencer::get_spring_anchor_vars_for_spring, py::arg("springIndex"))
        .def("getSpringAnchors", &Tencer::get_spring_anchors)
        .def("getSpringAnchorsForSpring", &Tencer::get_spring_anchors_for_spring, py::arg("springIndex"))
        .def("getRestMaterialVarsOpenRods", &Tencer::get_rest_material_vars_open_rods)
        .def("getRestMaterialVarsClosedRods", &Tencer::get_rest_material_vars_closed_rods)
        .def("getRestMaterialVarsForRod", &Tencer::get_rest_material_vars_for_rod, py::arg("rodIndex"))

        // rest variable settings
        .def("getRestVarsType", &Tencer::get_rest_vars_type)
        .def("setRestVarsType", &Tencer::set_rest_vars_type, py::arg("rvt"), py::arg("updateState") = true)

        // energy and gradients
        .def("energy", py::overload_cast<KnotEnergyType,EnergyType> (&Tencer::energy, py::const_), py::arg("knotEnergyType"), py::arg("energyType"))
        .def("gradient", py::overload_cast<KnotEnergyType, EnergyType, VariableMask, bool>(&Tencer::gradient,  py::const_), py::arg("knotEnergyType"), py::arg("energyType"),py::arg("vmask"),py::arg("updateParam") = false)
        .def("hessian", &Tencer::get_hessian, py::arg("vmask") = VariableMask::Defo, py::arg("knotEnergyType") = KnotEnergyType::Full, py::arg("energyType") = EnergyType::Full)
        .def("hessianSparsityPattern",py::overload_cast<KnotEnergyType,VariableMask,Real>(&Tencer::hessianSparsityPattern, py::const_), py::arg("knotEnergyType"), py::arg("vmask"), py::arg("val") = 0.0)
        .def("applyHessian", &Tencer::apply_hessian, py::arg("direction"), py::arg("mask") = HessianComputationMask())

        .def("massMatrix",py::overload_cast<>(&Tencer::massMatrix, py::const_))

       
        // pickling
        // TODO
        // .def(py::pickle([](const Tencer &knot) {
        //         return py::make_tuple(
        //             knot.get_open_rods(), 
        //             knot.get_closed_rods(), 
        //             knot.get_springs(), 
        //             knot.get_attachment_vertices(), 
        //             knot.get_target_rods()
        //         );
        //     },
        //     [](const py::tuple &t) {
        //         if (t.size() != 6) throw std::runtime_error("Invalid state!");
        //         std::vector<ElasticRod> open_rods = t[0].cast<std::vector<ElasticRod>>();
        //         std::vector<PeriodicRod> closed_rods = t[1].cast<std::vector<PeriodicRod>>();
        //         std::vector<Spring> springs = t[2].cast<std::vector<Spring>>();
        //         std::vector<SpringAttachmentVertices> attachment_vertices = t[3].cast<std::vector<SpringAttachmentVertices>>();
        //         std::vector<ElasticRod> target_rods = t[5].cast<std::vector<ElasticRod>>();
        //         return Tencer(open_rods, closed_rods, springs, attachment_vertices, target_rods);
        //     }))
    ;


    py::class_<OO, std::shared_ptr<OO>>(m, "OptimizationObject")

        //Objective function and derivatives
        .def("J", py::overload_cast<>(&OO::J))
        .def("J", py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&OO::J), py::arg("params"), py::arg("opt_etype") = OptEnergyType::Full)
        .def("gradp_J", &OO::gradp_J, py::arg("design_params"), py::arg("eType") = EnergyType::Full)
        .def("apply_hessian", &OO::apply_hess, py::arg("params"), py::arg("deltap"), py::arg("coeffJ") = 1.0, py::arg("eType") = EnergyType::Full)

        // Accessors
        .def("numParams", &OO::numParams)
        .def("params", &OO::params)

        .def("newPt", &OO::newPt, py::arg("params"))
        ;

    py::class_<LOO, std::shared_ptr<LOO>>(m, "OptimizationObjectWithLinearChangeOfVariable")

        //Objective function and derivatives
        .def("J", py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&LOO::J), py::arg("params"), py::arg("opt_etype") = OptEnergyType::Full)
        .def("gradp_J", &LOO::gradp_J, py::arg("design_params"), py::arg("eType") = EnergyType::Full)
        .def("apply_hessian", &LOO::apply_hess, py::arg("params"), py::arg("deltap"), py::arg("coeffJ") = 1.0, py::arg("eType") = EnergyType::Full)

        // Accessors
        .def("numParams", &LOO::numParams)
        .def("params", &LOO::params)
        
        .def("newPt", &OO::newPt, py::arg("params"))

        // Transformations
        .def("applyTransformation", &LOO::apply_transformation)
        ;

    py::class_<EOO, std::shared_ptr<EOO>>(m, "EquilibriumOptimizationObject")

        //Objective function and derivatives
        .def("J", py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&EOO::J), py::arg("params"), py::arg("opt_etype") = OptEnergyType::Full)
        .def("gradp_J", &EOO::gradp_J, py::arg("design_params"), py::arg("eType") = EnergyType::Full)
        .def("apply_hessian", &EOO::apply_hess, py::arg("params"), py::arg("deltap"), py::arg("coeffJ") = 1.0, py::arg("eType") = EnergyType::Full)

        // Accessors
        .def("numParams", &EOO::numParams)
        .def("params", &EOO::params)
        .def("linesearchObject", &EOO::linesearchObject)
        .def("committedObject", &EOO::committedObject)
        .def("linesearchWorkingSet", &EOO::linesearchWorkingSet)
        .def("committedWorkingSet", &EOO::committedWorkingSet)
        .def_readonly("objective", &EOO::objective, py::return_value_policy::reference)

        .def("commitLineseachObject", &EOO::commitLinesearchObject)
        .def("invalidateAdjointState", &EOO::invalidateAdjointState)
        .def("invalidateEquilibria", &EOO::invalidateEquilibria)

        .def("newPt", &OO::newPt, py::arg("params"))
        ;





    py::class_<WeightedTargetFittingDOOT, std::shared_ptr<WeightedTargetFittingDOOT>>(m, "WeightedTargetFittingDOOT")
        .def(py::init<Tencer &, std::vector<ElasticRod> &, std::vector<PeriodicRod> &, std::vector<VecX> &>(), py::arg("tencer"), py::arg("open_target_rods"), py::arg("closed_target_rods"), py::arg("radii"))
        .def("object", &WeightedTargetFittingDOOT::object)
        .def("computeValue", &WeightedTargetFittingDOOT::computeValue)
        .def("computeGrad", &WeightedTargetFittingDOOT::computeGrad)
        .def("computeDeltaGrad", &WeightedTargetFittingDOOT::computeDeltaGrad, py::arg("delta_xp"))
        ;


    using PyMRSP = py::class_<TencerOptimization, EOO, std::shared_ptr<TencerOptimization>>;
    auto spring_sparsifier = PyMRSP(m, "TencerOptimization");

    spring_sparsifier
        .def(py::init<Tencer &, const NewtonOptimizerOptions &, const std::vector<size_t> &, std::vector<ElasticRod> &, std::vector<PeriodicRod> &, std::vector<VecX> &, Eigen::VectorXd, Real, PredictionOrder>(), py::arg("tencer"), py::arg("Newton_optimizer_options"), py::arg("fixed_vars"), py::arg("open_target_rod"), py::arg("closed_target_rod"), py::arg("radii"), py::arg("sparsification_weights") = Eigen::VectorXd(), py::arg("hessian_shift") = 1e-6, py::arg("prediction_order") = PredictionOrder::One)

        // Objective function and derivatives
        .def("J", py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&TencerOptimization::J), py::arg("params"), py::arg("opt_etype") = OptEnergyType::Full)
        .def("gradp_J", &TencerOptimization::gradp_J, py::arg("design_params"), py::arg("eType") = OptEnergyType::Full)
        .def("apply_hessian", &TencerOptimization::apply_hess, py::arg("params"), py::arg("deltap"), py::arg("coeffJ") = 1.0, py::arg("eType") = OptEnergyType::Full)

        .def("update_target_rods", &TencerOptimization::update_target_rods, py::arg("open_rods"), py::arg("closed_rods"), py::arg("radii"))

        // Accessors
        .def("numParams", &TencerOptimization::numParams)
        .def("params", &TencerOptimization::params)
        .def("linesearchObject", &TencerOptimization::linesearchObject)
        .def("committedObject", &TencerOptimization::committedObject)
        .def("linesearchWorkingSet", &TencerOptimization::linesearchWorkingSet)
        .def("committedWorkingSet", &TencerOptimization::committedWorkingSet)
        .def_readonly("objective", &TencerOptimization::objective, py::return_value_policy::reference)
        .def("set_norm_param", &TencerOptimization::set_norm_param, py::arg("param"))

        .def("commitLineseachObject", &TencerOptimization::commitLinesearchObject)
        .def("invalidateAdjointState", &TencerOptimization::invalidateAdjointState)
        .def("invalidateEquilibria", &TencerOptimization::invalidateEquilibria)

        .def("newPt", &OO::newPt, py::arg("params"))

        // Bounds
        .def("getVarsLowerBounds", &TencerOptimization::get_vars_lower_bounds)
        .def("getVarsUpperBounds", &TencerOptimization::get_vars_upper_bounds)

    ;


    using PySymSp = py::class_<TencerOptimizationSymmetries, LOO, std::shared_ptr<TencerOptimizationSymmetries>>;
    auto symmetry_sparsifier = PySymSp(m, "TencerOptimizationSymmetries");

    symmetry_sparsifier
        .def(py::init<CSCMatrix<SuiteSparse_long, Real> &, Tencer &, const NewtonOptimizerOptions &, const std::vector<size_t> &, std::vector<ElasticRod> &, std::vector<PeriodicRod> &, std::vector<VecX> &, VecX , const VecX &, Real, PredictionOrder>(), py::arg("symmetry_groups"), py::arg("tencer"), py::arg("Newton_optimizer_options"), py::arg("fixed_vars"), py::arg("open_target_rod"), py::arg("closed_target_rod"), py::arg("radii"), py::arg("sparsification_weights") = Eigen::VectorXd(), py::arg("length_offset") = std::vector<std::vector<Real>>(), py::arg("hessian_shift") = 1e-6, py::arg("prediction_order") = PredictionOrder::One)

        .def("update_target_rods", &TencerOptimizationSymmetries::update_target_rods, py::arg("open_rods"), py::arg("closed_rods"), py::arg("radii"))


        // Objective function and derivatives
        .def("J", py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&TencerOptimizationSymmetries::J), py::arg("params"), py::arg("opt_etype") = OptEnergyType::Full)
        .def("gradp_J", &TencerOptimizationSymmetries::gradp_J, py::arg("design_params"), py::arg("eType") = OptEnergyType::Full)
        .def("apply_hessian", &TencerOptimizationSymmetries::apply_hess, py::arg("params"), py::arg("deltap"), py::arg("coeffJ") = 1.0, py::arg("eType") = OptEnergyType::Full)

        // Accessors
        .def("numParams", &TencerOptimizationSymmetries::numParams)
        .def("params", &TencerOptimizationSymmetries::params)

        .def("newPt", &OO::newPt, py::arg("params"))

        // Transformations
        .def("applyTransformation", &TencerOptimizationSymmetries::apply_transformation)
        .def("applyTransformationAffine", &TencerOptimizationSymmetries::apply_transformation_affine)

        // Bounds
        .def("getVarsLowerBounds", &TencerOptimizationSymmetries::get_vars_lower_bounds)
        .def("getVarsUpperBounds", &TencerOptimizationSymmetries::get_vars_upper_bounds)

    ;



    ////////////////////////////////////////////////////////////////////////////////
    // Knitro Optimization
    ////////////////////////////////////////////////////////////////////////////////
#if HAS_KNITRO


    m.def("optimize",
        [](TencerOptimization &tk_opt, OptAlgorithm alg, size_t num_steps,
              Real trust_region_scale, Real optimality_tol, std::function<void()> &update_viewer,std::vector<int> & fixed_rest_vars) {
            return optimize(tk_opt, alg, num_steps, trust_region_scale, optimality_tol, update_viewer, fixed_rest_vars);
        },
        py::arg("spring_sparsifier"), 
        py::arg("alg"), 
        py::arg("num_steps"), 
        py::arg("trust_region_scale"), 
        py::arg("optimality_tol"), 
        py::arg("update_viewer"),
        py::arg("fixed_rest_vars") = std::vector<int>()
    );


    m.def("optimizeWithSymmetries",
        [](TencerOptimizationSymmetries &tk_opt, OptAlgorithm alg, size_t num_steps,
              Real trust_region_scale, Real optimality_tol, std::function<void()> &update_viewer, std::vector<int> & fixed_rest_vars) {
            return optimize(tk_opt, alg, num_steps, trust_region_scale, optimality_tol, update_viewer, fixed_rest_vars);
        },
        py::arg("symmetry_sparsification"), 
        py::arg("alg"), 
        py::arg("num_steps"), 
        py::arg("trust_region_scale"), 
        py::arg("optimality_tol"), 
        py::arg("update_viewer"),
        py::arg("fixed_rest_vars") = std::vector<int>()
    );

#endif


//     // ----------------------------------------------------------------------------
//     //                                 Utilities
//     // ----------------------------------------------------------------------------

    m.def("minimize_twist",
        [](PeriodicRod &pr, bool verbose) {
            py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
            py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
            return minimize_twist(pr, verbose);
        },
        py::arg("pr"),
        py::arg("verbose") = false
    );

    m.def("spread_twist_preserving_link",
        [](PeriodicRod &pr, bool verbose) {
            py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
            py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
            return spread_twist_preserving_link(pr, verbose);
        },
        py::arg("pr"),
        py::arg("verbose") = false
    );

    

    m.def("computeEquilibrium",
    [](Tencer &obj, const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &opts, PyCallbackFunction pcb, Real systemEnergyIncreaseFactorLimit, Real energyLimitingThreshold, Real hessianShift) {
            return compute_equilibrium(obj, fixedVars, opts, callbackWrapper(pcb), systemEnergyIncreaseFactorLimit, energyLimitingThreshold, hessianShift);
        },
        py::arg("tencer"),
        py::arg("fixedVars") = std::vector<size_t>(), 
        py::arg("opts") = NewtonOptimizerOptions(), py::arg("cb") = nullptr,
        py::arg("systemEnergyIncreaseFactorLimit") = safe_numeric_limits<Real>::max(), 
        py::arg("energyLimitingThreshold") = 1e-6,
        py::arg("hessianShift") = 0.0,
        py::call_guard<py::scoped_ostream_redirect,
                        py::scoped_estream_redirect>());
    



}