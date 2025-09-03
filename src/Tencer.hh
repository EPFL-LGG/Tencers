#ifndef Tencers_HH
#define Tencers_HH

#include <ElasticRods/PeriodicRod.hh>
#include "helpers.hh"
#include "Spring.hh"
#include "SlidingNode.hh"

#include <unordered_set>  // to find sliding nodes' overlapping stencils

template<typename Real_>
struct Tencer_T;

struct SpringAttachmentVertices {

    SpringAttachmentVertices(int rA, int vA, int rB, int vB) : rod_idx_A(rA), vertex_A(vA), rod_idx_B(rB), vertex_B(vB) {}
    SpringAttachmentVertices() = default;

    size_t rod_idx_A;
    size_t vertex_A;
    size_t rod_idx_B;
    size_t vertex_B;

    // Comparison operator
    bool operator==(const SpringAttachmentVertices &sav) const {
        return rod_idx_A == sav.rod_idx_A && vertex_A == sav.vertex_A && rod_idx_B == sav.rod_idx_B && vertex_B == sav.vertex_B;
    }
};

template<typename Real_>
struct SpringAnchorVars_T {
    int   rod_idx_A;
    Real_ mat_coord_A;
    int   rod_idx_B;
    Real_ mat_coord_B;

    SpringAnchorVars_T(int rA, Real_ mA, int rB, Real_ mB) : rod_idx_A(rA), mat_coord_A(mA), rod_idx_B(rB), mat_coord_B(mB) {}
    SpringAnchorVars_T() = default;

    // Copy constructor
    template<typename Real_2>
    SpringAnchorVars_T(const SpringAnchorVars_T<Real_2> &sav) {
        rod_idx_A = sav.rod_idx_A;
        mat_coord_A = sav.mat_coord_A;
        rod_idx_B = sav.rod_idx_B;
        mat_coord_B = sav.mat_coord_B;
    }

    // Comparison operator
    bool operator==(const SpringAnchorVars_T &sav) const {
        Real_ tol = 1e-10;
        return rod_idx_A == sav.rod_idx_A && std::abs(mat_coord_A - sav.mat_coord_A) < tol && rod_idx_B == sav.rod_idx_B && std::abs(mat_coord_B - sav.mat_coord_B) < tol;
    }

    // Use 0-1 notation in place of A-B hardcodings
    Real_ &operator[](size_t i)         { if (i == 0) return mat_coord_A; else if (i == 1) return mat_coord_B; else throw std::runtime_error("Invalid index");}
    int    getRodIdx (size_t i) const   { if (i == 0) return   rod_idx_A; else if (i == 1) return   rod_idx_B; else throw std::runtime_error("Invalid index");}
    void   setRodIdx (size_t i, int ri) { if (i == 0) rod_idx_A = ri;     else if (i == 1) rod_idx_B = ri;     else throw std::runtime_error("Invalid index");}
};

enum class VariableMask { Defo, Rest, All };

template<typename Real_>
struct Tencer_T {

    using Vec3_T = Eigen::Matrix<Real_, 3, 1>;
    using VecX_T  = Eigen::Matrix<Real_, Eigen::Dynamic, 1>;
    using CSCMat_T = CSCMatrix<SuiteSparse_long, Real_>;
    using MatX_T = Eigen::Matrix<Real_, -1, -1>;
    using TMatrix = TripletMatrix<Triplet<Real_>>;

    using PR = PeriodicRod_T<Real_>;
    using EnergyType = typename PR::EnergyType;

    using SlidingNode = SlidingNode_T<Real_>;
    using SpringAnchorVars = SpringAnchorVars_T<Real_>;
    using SpringAnchors = std::array<std::shared_ptr<SlidingNode>, 2>;
    using Spring = Spring_T<Real_>;
    using RestVarsType = typename Spring::RestVarsType;

    enum class KnotEnergyType {Full, Elastic, Springs};

    std::vector<ElasticRod_T<Real_>> open_rods;
    std::vector<PeriodicRod_T<Real_>> closed_rods;
    std::vector<Spring_T<Real_>> springs;
    std::vector<ElasticRod> target_rods;
    bool sparsityPatternUpdated = true; // used to update sparsity pattern between every equilibrium computation with sliding nodes

    // Constructor from rod list and springs
    Tencer_T(const std::vector<ElasticRod_T<Real_>> &er, const std::vector<PeriodicRod_T<Real_>> &pr, const std::vector<Spring_T<Real_>> &sp, std::vector<SpringAttachmentVertices> &v, std::vector<ElasticRod> &targets);

    // Copy another tencer.
    template<typename Real_2>
    void set(const Tencer_T<Real_2> &k) {
        castStdADVector(k.open_rods, open_rods);
        castStdADVector(k.closed_rods, closed_rods);

        // Set rest_vars_type
        if (k.get_rest_vars_type() == Tencer_T<Real_2>::RestVarsType::StiffnessOnVerticesFast)
            rest_vars_type = RestVarsType::StiffnessOnVerticesFast;
        else if (k.get_rest_vars_type() == Tencer_T<Real_2>::RestVarsType::Stiffness)
            rest_vars_type = RestVarsType::Stiffness;
        else if (k.get_rest_vars_type() == Tencer_T<Real_2>::RestVarsType::SpringAnchors)
            rest_vars_type = RestVarsType::SpringAnchors;
        else if (k.get_rest_vars_type() == Tencer_T<Real_2>::RestVarsType::StiffnessAndSpringAnchors)
            rest_vars_type = RestVarsType::StiffnessAndSpringAnchors;
        else
            throw std::runtime_error("Unknown rest_vars_type");

        initialize_rest_material_vars();
        castStdADVector(k.springs,springs);
        m_defoVarsIndexPerRod = k.get_defo_vars_index_per_rod();
        m_defoVarsPerRod = k.get_defo_vars_per_rod();
        m_numDefoVars = k.numDefoVars();
        castStdADVectorOfVectors(k.get_rest_material_vars_open_rods(), rest_mat_vars_open_rods);
        castStdADVectorOfVectors(k.get_rest_material_vars_closed_rods(), rest_mat_vars_closed_rods);

        m_attachment_vertices = k.get_attachment_vertices();   // std::vector, safe to copy in any case, even if it is empty e.g. because we are using SpringAnchors
        if (rest_vars_type == RestVarsType::Stiffness || rest_vars_type == RestVarsType::SpringAnchors || rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
            castStdADVector(k.get_spring_anchor_vars(), spring_anchor_vars);

            spring_anchors.resize(springs.size());
            for (size_t si = 0; si < springs.size(); ++si) {
                for (size_t ai = 0; ai < 2; ++ai) {
                    
                    if (k.get_spring_anchors_for_spring(si)[ai]->type() == "LinearClosedSlidingNode")
                        spring_anchors[si][ai] = std::make_shared<LinearClosedSlidingNode_T<Real_>>(*std::dynamic_pointer_cast<LinearClosedSlidingNode_T<Real_2>>(k.get_spring_anchors_for_spring(si)[ai]));
                    else if (k.get_spring_anchors_for_spring(si)[ai]->type() == "LinearOpenSlidingNode")
                        spring_anchors[si][ai] = std::make_shared<LinearOpenSlidingNode_T<Real_>>(*std::dynamic_pointer_cast<LinearOpenSlidingNode_T<Real_2>>(k.get_spring_anchors_for_spring(si)[ai]));
                    else
                        throw std::runtime_error("Unknown node type");
                    
                    size_t ri = spring_anchor_vars[si].getRodIdx(ai);
                    std::shared_ptr<std::vector<Vec3_T>> points = std::make_shared<std::vector<Vec3_T>>(get_deformed_points_for_rod(ri));
                    std::shared_ptr<std::vector<Real_>> rest_mat_vars = std::make_shared<std::vector<Real_>>(get_rest_material_vars_for_rod(ri));
                    spring_anchors[si][ai]->set_trajectory(points);
                    spring_anchors[si][ai]->set_rest_material_vars(rest_mat_vars);
                    spring_anchors[si][ai]->set(spring_anchor_vars[si][ai]);
                }
            }
        }

        set_rest_vars_type(rest_vars_type);
        update_spring_endpoint_coords();
    }
    
    // Copy constructor converting from another floating point type (e.g., double to autodiff)
    template<typename Real_2>
    Tencer_T(const Tencer_T<Real_2> &k) { set(k); }

    // Copy constructor
    Tencer_T(const Tencer_T &k) { set(k); }

    // Copy assignment converting from another floating point type (e.g., double to autodiff)
    template<typename Real_2>
    Tencer_T<Real_>& operator=(const Tencer_T<Real_2> &r) { 
        set(r); 
        return *this;
    }

    // Copy assignment
    Tencer_T& operator=(const Tencer_T &r) { 
        set(r); 
        return *this;
    }

    std::string mangledName() const {
        return "Tencer" + autodiffOrNotString<Real_>() + ">"; 
    }

    // update springs' coordinates
    Vec3_T find_vertex_coords(size_t rod_idx, int num_vertex);
    void update_spring_endpoint_coords();
    void initialize_spring_anchors();
    void initialize_rest_material_vars();
    std::vector<Real_> compute_rest_material_vars_open_rod  (const ElasticRod_T<Real_>  rod);
    std::vector<Real_> compute_rest_material_vars_closed_rod(const PeriodicRod_T<Real_> rod);
    void set_spring_anchor_vars(const std::vector<SpringAnchorVars> &sav) { set_spring_anchors(sav); }  // alias
    void set_spring_anchors(const std::vector<SpringAnchorVars> &sav);
    void set_spring_anchors(const Eigen::Ref<const VecX_T>      &sav);

    RestVarsType get_rest_vars_type() const { return rest_vars_type; }
    void set_rest_vars_type(RestVarsType rvt, bool updateState = true);

    std::vector<Real> get_rest_vars_upper_bounds() const;
    std::vector<Real> get_rest_vars_lower_bounds() const;
    std::vector<Real> get_spring_anchors_upper_bounds() const;
    std::vector<Real> get_spring_anchors_lower_bounds() const;

    // getters : rods and springs
    std::vector<ElasticRod_T<Real_>> get_open_rods() const {return open_rods;}
    std::vector<PeriodicRod_T<Real_>> get_closed_rods() const {return closed_rods;}
    std::vector<Spring_T<Real_>> get_springs() const {return springs;}
    Spring_T<Real_> get_spring(int i) const {return springs[i];}
    size_t num_open_rods() const {return open_rods.size();}
    size_t num_closed_rods() const {return closed_rods.size();}
    size_t num_rods() const {return open_rods.size() + closed_rods.size();}
    size_t num_springs() const {return springs.size();}
    VecX_T springs_coord_diff() const;
    std::vector<ElasticRod> get_target_rods() const {return target_rods;};
    
    // getters ans setters : attributes
    std::vector<SpringAttachmentVertices> get_attachment_vertices() const {return m_attachment_vertices;}
    SpringAttachmentVertices get_attachment_vertices_for_spring(int i) const {return m_attachment_vertices[i];}
    std::vector<SpringAnchorVars> get_spring_anchor_vars() const {return spring_anchor_vars;}
    SpringAnchorVars get_spring_anchor_vars_for_spring(int i) const {return spring_anchor_vars[i];}
    std::vector<SpringAnchors> get_spring_anchors() const {return spring_anchors;}
    SpringAnchors get_spring_anchors_for_spring(int i) const {return spring_anchors[i];}
    const std::vector<std::vector<Real_>> &get_rest_material_vars_open_rods() const {return rest_mat_vars_open_rods;}
    const std::vector<std::vector<Real_>> &get_rest_material_vars_closed_rods() const {return rest_mat_vars_closed_rods;}
    const std::vector<Real_> &get_rest_material_vars_for_rod(size_t rod_idx) const;
    const std::vector<Pt3_T<Real_>> get_deformed_points_for_rod(size_t rod_idx) const;

    // getters and setters : defo and rest variables
    size_t numDefoVars() const {return m_numDefoVars;}
    size_t numRestVars() const;
    size_t numVars(VariableMask vmask = VariableMask::Defo) const;
    VecX_T getDefoVars() const; 
    VecX_T getRestVars() const;
    VecX_T getVars(VariableMask vmask = VariableMask::Defo) const;
    void setDefoVars(const Eigen::Ref<const VecX_T> &vars){m_setDefoVars(vars);}
    void setRestVars(const Eigen::Ref<const VecX_T> &vars){m_setRestVars(vars);}
    void setVars(const Eigen::Ref<const VecX_T> &vars,VariableMask vmask = VariableMask::Defo);
    size_t restVarsOffset() const {return numDefoVars();}
    size_t kVarsOffset(RestVarsType rvt, VariableMask vmask) const;
    size_t uVarsOffset(RestVarsType rvt, VariableMask vmask) const;
    size_t get_defo_var_index_for_rod(size_t i) const {return m_defoVarsIndexPerRod[i];}
    size_t get_num_defo_var_for_rod(size_t i) const {return m_defoVarsPerRod[i];}
    std::vector<size_t> get_defo_vars_per_rod() const {return m_defoVarsPerRod;}
    std::vector<size_t> get_defo_vars_index_per_rod() const {return m_defoVarsIndexPerRod;}



    // Energies, gradients, hessians
    
    Real_ energy() const;
    Real_ energy(KnotEnergyType ke,EnergyType e = EnergyType::Full) const;
    Real_ elastic_energy(EnergyType etype = EnergyType::Full) const;
    Real_ springs_energy() const;
    
    VecX_T gradient(bool updatedParametrization = false, VariableMask vmask = VariableMask::Defo) const;
    VecX_T gradient(KnotEnergyType ke, EnergyType e, VariableMask vmask = VariableMask::Defo, bool updatedParametrization = false) const;
    VecX_T rods_gradient(bool updatedParametrization = false, VariableMask vmask = VariableMask::Defo, EnergyType e = EnergyType::Full) const;
    VecX_T rods_gradient_defo(bool updatedParametrization = false, EnergyType e = EnergyType::Full) const;
    VecX_T springs_gradient(VariableMask vmask = VariableMask::Defo) const;
    VecX_T springs_gradient_defo() const;
    VecX_T springs_gradient_rest() const;
    
    void hessian(CSCMat_T &Hout, bool /*projectionMask*/ = false, VariableMask vmask = VariableMask::Defo) const;
    void hessian(CSCMat_T &Hout, VariableMask vmask = VariableMask::Defo, KnotEnergyType ke = KnotEnergyType::Full, EnergyType e = EnergyType::Full) const;
    void rods_hessian(CSCMat_T &Hout, VariableMask vmask = VariableMask::Defo, EnergyType e = EnergyType::Full) const;
    void springs_hessian(CSCMat_T &Hout, VariableMask vmask = VariableMask::Defo) const;
    CSCMat_T get_hessian(VariableMask vmask = VariableMask::Defo, KnotEnergyType ke = KnotEnergyType::Full, EnergyType e = EnergyType::Full) const;
    void rods_hessian_defo(size_t size, CSCMat_T &Hout, EnergyType e = EnergyType::Full) const;
    void springs_hessian_defo(CSCMat_T &Hout) const;
    void springs_hessian_dxdk(CSCMat_T &Hout) const;
    void springs_hessian_dxdu(CSCMat_T &Hout) const;
    void springs_hessian_du2(CSCMat_T &Hout, VariableMask vmask = VariableMask::Rest) const;
    void springs_hessian_dkdu(CSCMat_T &Hout, VariableMask vmask = VariableMask::Rest) const;
    
    CSCMat_T hessianSparsityPattern(Real_ val = 0.0, VariableMask vmask = VariableMask::Defo) const;
    CSCMat_T hessianSparsityPattern(KnotEnergyType ke,VariableMask vmask = VariableMask::Defo,Real_ val = 0.0) const;
    CSCMat_T rods_hessianSparsityPattern(Real_ val = 0.0, VariableMask vmask = VariableMask::Defo) const;
    CSCMat_T rods_hessianSparsityPattern_defo(Real_ val = 0.0) const;
    CSCMat_T springs_hessianSparsityPattern(Real_ val = 0.0, VariableMask vmask = VariableMask::Defo) const;
    CSCMat_T springs_hessianSparsityPattern_defo(size_t size, Real_ val = 0.0) const;
    CSCMat_T springs_hessianSparsityPattern_dxdk(Real_ val = 0.0) const;
    CSCMat_T springs_hessianSparsityPattern_dxdu(Real_ val = 0.0) const;
    CSCMat_T springs_hessianSparsityPattern_du2(Real_ val = 0.0, VariableMask vmask = VariableMask::Rest) const;
    CSCMat_T springs_hessianSparsityPattern_dkdu(Real_ val = 0.0, VariableMask vmask = VariableMask::Rest) const;

    VecX_T apply_hessian(const VecX_T &v, const HessianComputationMask &mask = HessianComputationMask());
    void apply_rods_hessian_defo_defo(const VecX_T &v, VecX_T &res, const HessianComputationMask &mask);
    void apply_springs_hessian_defo_defo(const VecX_T &v, VecX_T &res);
    void apply_springs_hessian_defo_rest(const VecX_T &v, VecX_T &res);
    void apply_springs_hessian_rest_defo(const VecX_T &v, VecX_T &res);
    void apply_springs_hessian_rest_rest(const VecX_T &v, VecX_T &res);

    // Hessian methods required by the DesignOptimization framework to simplify computations
    VecX_T apply_hessian_defoVarsIn_restVarsOut(const VecX_T &v);
    VecX_T apply_hessian_defoVarsOut_restVarsIn(const VecX_T &v);
    VecX_T apply_hessian_defoVars_restVarsOut(const VecX_T &v);


    // Additional methods required by compute_equilibrium
    void updateSourceFrame()              { for(auto &r : open_rods) r.updateSourceFrame(); for(auto &r : closed_rods) r.updateSourceFrame();}
    void updateRotationParametrizations() { for(auto &r : open_rods) r.updateRotationParametrizations(); for(auto &r : closed_rods) r.updateRotationParametrizations();}
    // Update our parametrization of the system's DoFs for comptibility with EquilibriumSolver (MeshFEM)
    void updateParametrization() {updateSourceFrame(); updateRotationParametrizations();}
    Real characteristicLength() const { return  1.0; }
    Real approxLinfVelocity(const VecX_T & /* d */) const { return -1.0; }
    void massMatrix(CSCMat_T &M, bool /* updatedParametrization */, bool /* lumped */) const { M.setIdentity(true); }
    TMatrix massMatrix() const {
        auto M = hessianSparsityPattern(false);
        massMatrix(M,true,true);
        return M.getTripletMatrix();
    }


    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Derived::ColsAtCompileTime>
    visualizationField(const std::vector<Derived> &perRodFields) const {
        if (perRodFields.size() != num_rods()) throw std::runtime_error("Invalid per-rod-field size");
        using FieldStorage = Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Derived::ColsAtCompileTime>;
        std::vector<FieldStorage> perRodVisualizationFields;
        perRodVisualizationFields.reserve(num_rods());
        int fullSize = 0;
        const int cols = perRodFields.at(0).cols();
        for (size_t ri = 0; ri < num_closed_rods(); ++ri) {
            if (cols != perRodFields[ri].cols()) throw std::runtime_error("Mixed field types forbidden.");
            perRodVisualizationFields.push_back((get_closed_rods()[ri]).rod.visualizationField(perRodFields[ri]));
            fullSize += perRodVisualizationFields.back().rows();
        }
        FieldStorage result(fullSize, cols);
        int offset = 0;
        for (const auto &vf : perRodVisualizationFields) {
            result.block(offset, 0, vf.rows(), cols) = vf;
            offset += vf.rows();
        }
        assert(offset == fullSize);
        return result;
    }


    

protected:
    std::vector<std::vector<Real_>> rest_mat_vars_open_rods;     // The location of the rod nodes in the material domain (i.e. the cumulated sum of the edge rest lengths) for open elastic rods
    std::vector<std::vector<Real_>> rest_mat_vars_closed_rods;   // The location of the rod nodes in the material domain (i.e. the cumulated sum of the edge rest lengths) for peroidic elastic rods
    std::vector<SpringAnchorVars> spring_anchor_vars;            // Material location of the spring anchors (used when rest_vars_type != RestVarsType::StiffnessOnVerticesFast)
    std::vector<SpringAnchors> spring_anchors;                   // Spatial  location of the spring anchors (used when rest_vars_type != RestVarsType::StiffnessOnVerticesFast)
    std::vector<SpringAttachmentVertices> m_attachment_vertices; // The indices of the rod nodes to which springs are attached (used when rest_vars_type == RestVarsType::StiffnessOnVerticesFast)
    std::vector<size_t> m_defoVarsIndexPerRod;
    std::vector<size_t> m_defoVarsPerRod;
    size_t m_numDefoVars;
    
    RestVarsType rest_vars_type = RestVarsType::StiffnessOnVerticesFast;

    // Inherited protected methods 
    void m_setDefoVars(const Eigen::Ref<const VecX_T> &vars);
    void m_setRestVars(const Eigen::Ref<const VecX_T> &vars);

};




#endif