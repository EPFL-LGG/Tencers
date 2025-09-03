#ifndef Spring_HH
#define Spring_HH
#include "SlidingNode.hh"

template <typename Real_>
struct  Spring_T {

    enum class RestVarsType {
        StiffnessOnVerticesFast,   // k only, with spring nodes on vertices; specialized implementation that can be faster than the more general Stiffness
        Stiffness,                 // k only, with spring nodes anywhere on edges (u does not necessarily match with any reference material variable)
        SpringAnchors,             // u only
        StiffnessAndSpringAnchors  // k and u
    };

    using Vec2_T = Eigen::Matrix<Real_, 2, 1>;
    using Vec3_T = Eigen::Matrix<Real_, 3, 1>;
    using Vec6_T = Eigen::Matrix<Real_, 6, 1>;
    using VecX_T  = Eigen::Matrix<Real_, Eigen::Dynamic, 1>;
    using CSCMat_T = CSCMatrix<SuiteSparse_long, Real_>;
    using Mat2_T = Eigen::Matrix<Real_, 2, 2>;
    using Mat3_T = Eigen::Matrix<Real_, 3, 3>;
    using MatX_T = Eigen::Matrix<Real_, -1, -1>;


    std::shared_ptr<Node_T<Real_>> position_A;
    std::shared_ptr<Node_T<Real_>> position_B;
    Real_ stiffness;
    double rest_length;

    Spring_T(Vec3_T &pA, Vec3_T &pB, Real_ s, double l = 0);
    Spring_T(const std::shared_ptr<Node_T<Real_>> &pA, const std::shared_ptr<Node_T<Real_>> &pB, Real_ s, double l = 0);

    // Copy constructor converting from another floating point type (e.g., double to autodiff)
    template<typename Real_2>
    Spring_T(const Spring_T<Real_2> &sp){ set(sp); }

    // Copy constructor
    Spring_T(const Spring_T &sp){ set(sp); }

    // Copy assignment converting from another floating point type (e.g., double to autodiff)
    template<typename Real_2>
    Spring_T<Real_>& operator=(const Spring_T<Real_2> &sp) { 
        set(sp);
        return *this;
    }

    // Copy assignment
    Spring_T& operator=(const Spring_T &sp) { 
        set(sp);
        return *this;
    }

    ~Spring_T() = default;

    std::string mangledName() const {
        return "Spring" + autodiffOrNotString<Real_>() + ">"; 
    }

    template<typename Real_2>
    void set(const Spring_T<Real_2> &sp){
        if (sp.position_A->type() == "FixedNode")
            position_A = std::make_shared<FixedNode_T<Real_>>(*std::dynamic_pointer_cast<FixedNode_T<Real_2>>(sp.position_A));
        else if (sp.position_A->type() == "LinearClosedSlidingNode")
            position_A = std::make_shared<LinearClosedSlidingNode_T<Real_>>(*std::dynamic_pointer_cast<LinearClosedSlidingNode_T<Real_2>>(sp.position_A));
        else if (sp.position_A->type() == "LinearOpenSlidingNode")
            position_A = std::make_shared<LinearOpenSlidingNode_T<Real_>>(*std::dynamic_pointer_cast<LinearOpenSlidingNode_T<Real_2>>(sp.position_A));
        else
            throw std::runtime_error("Spring::set(): Unknown node type");

        // We need to separetely check the type of the second node because it could be on a rod of a different type (open/closed)
        if (sp.position_B->type() == "FixedNode")
            position_B = std::make_shared<FixedNode_T<Real_>>(*std::dynamic_pointer_cast<FixedNode_T<Real_2>>(sp.position_B));
        else if (sp.position_B->type() == "LinearClosedSlidingNode")
            position_B = std::make_shared<LinearClosedSlidingNode_T<Real_>>(*std::dynamic_pointer_cast<LinearClosedSlidingNode_T<Real_2>>(sp.position_B));
        else if (sp.position_B->type() == "LinearOpenSlidingNode")
            position_B = std::make_shared<LinearOpenSlidingNode_T<Real_>>(*std::dynamic_pointer_cast<LinearOpenSlidingNode_T<Real_2>>(sp.position_B));
        else
            throw std::runtime_error("Spring::set(): Unknown node type");

        stiffness = sp.stiffness;
        rest_length = sp.rest_length;

        // Set rest_vars_type
        if (sp.get_rest_vars_type() == Spring_T<Real_2>::RestVarsType::StiffnessOnVerticesFast)
            rest_vars_type = RestVarsType::StiffnessOnVerticesFast;
        else if (sp.get_rest_vars_type() == Spring_T<Real_2>::RestVarsType::Stiffness)
            rest_vars_type = RestVarsType::Stiffness;
        else if (sp.get_rest_vars_type() == Spring_T<Real_2>::RestVarsType::SpringAnchors)
            rest_vars_type = RestVarsType::SpringAnchors;
        else if (sp.get_rest_vars_type() == Spring_T<Real_2>::RestVarsType::StiffnessAndSpringAnchors)
            rest_vars_type = RestVarsType::StiffnessAndSpringAnchors;
        else
            throw std::runtime_error("Unknown rest_vars_type");

        set_rest_vars_type(rest_vars_type);
    }

    Real_ energy() const;

    // Derivatives w.r.t. individual anchor points
    Vec3_T dE_dx   (size_t i)           const;
    Real_  dE_du   (size_t i)           const;
    Mat3_T d2E_dx2 (size_t i, size_t j) const;
    Real_  d2E_du2 (size_t i, size_t j) const;
    Vec3_T d2E_dxdk(size_t i)           const;
    Vec3_T d2E_dxdu(size_t i, size_t j) const;
    Real_  d2E_dkdu(size_t i)           const;

    Vec6_T dE_dx() const;
    Vec2_T dE_du() const;
    VecX_T dE_dk() const;
    VecX_T dE_dxk() const;
    VecX_T dE_dxu() const;
    VecX_T dE_dku() const;
    VecX_T dE_dxku() const;
    MatX_T d2E_dx2()  const;
    Mat2_T d2E_du2()  const;
    VecX_T d2E_dxdk() const;
    VecX_T d2E_dxdu() const;

    

    size_t numRestVars() const {
        if      (rest_vars_type == RestVarsType::StiffnessOnVerticesFast)   return 1;
        else if (rest_vars_type == RestVarsType::Stiffness)                 return 1;
        else if (rest_vars_type == RestVarsType::SpringAnchors)             return 2;
        else if (rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) return 3;
        else throw std::runtime_error("Unknown rest_vars_type");
    }

    void set_rest_vars_type(RestVarsType rvt, bool updateState = true);
    RestVarsType get_rest_vars_type() const { return rest_vars_type; }

    VecX_T get_coords() const;
    Vec3_T coords_diff() const;
    double get_rest_length() const {return rest_length;}
    bool has_zero_rest_length() const { return rest_length == 0.0; }

protected:

    RestVarsType rest_vars_type = RestVarsType::StiffnessOnVerticesFast;
};




#endif