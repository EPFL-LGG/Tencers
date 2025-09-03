#include "Tencer.hh"
#include "helpers.hh"

// Constructor
template<typename Real_>
Tencer_T<Real_>::Tencer_T(const std::vector<ElasticRod_T<Real_>> &er, const std::vector<PeriodicRod_T<Real_>> &pr, const std::vector<Spring_T<Real_>> &sp, std::vector<SpringAttachmentVertices> &v,  std::vector<ElasticRod> &targets) : 
m_attachment_vertices(v) { 
    
    // Rods
    size_t idx = 0;
    for (auto r : er){
        open_rods.push_back(r);
        m_defoVarsIndexPerRod.push_back(idx);
        m_defoVarsPerRod.push_back(r.numDoF());
        idx += r.numDoF();
    }
    for (auto r : pr){
        closed_rods.push_back(r);
        m_defoVarsIndexPerRod.push_back(idx);
        m_defoVarsPerRod.push_back(r.numDoF());
        idx += r.numDoF();
    }
    m_numDefoVars = idx;

    // Springs
    if ((sp.size() != v.size())) {
        throw std::runtime_error("The number of attachment vertices does not match the number of springs!");
    }
    for (auto s : sp){
        springs.push_back(s);
    }
    initialize_rest_material_vars();
    update_spring_endpoint_coords();

    // Target rods
    for (auto tr : targets){
        target_rods.push_back(tr);
    }

}



// Given a vertex index in the rod, get the vertex coordinates
template<typename Real_>
Vec3_T<Real_> Tencer_T<Real_>::find_vertex_coords(size_t rod_idx, int num_vertex){
    VecX_T vars;
    Vec3_T coords = Vec3_T();
    if (rod_idx < open_rods.size()) vars = open_rods[rod_idx].getDoFs();
    else vars = closed_rods[rod_idx - open_rods.size()].getDoFs();
    for (int i = 0; i < 3; ++i){
        coords[i] = vars[3*num_vertex + i];
    }
    return coords;
}

// Update springs endpoint coordinates
template<typename Real_>
void Tencer_T<Real_>::update_spring_endpoint_coords(){
    BENCHMARK_START_TIMER_SECTION("Tencer update_spring_endpoint_coords");
    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast){
        for (size_t i = 0; i < springs.size(); ++i){
            if (m_attachment_vertices[i].vertex_A  == m_attachment_vertices[i].vertex_B && 
                m_attachment_vertices[i].rod_idx_A == m_attachment_vertices[i].rod_idx_B)
                throw std::runtime_error("Spring " + std::to_string(i) + " endpoints are both attached to vertex " + std::to_string(m_attachment_vertices[i].vertex_A) + " of rod " + std::to_string(m_attachment_vertices[i].rod_idx_A) + ". This is not supported in StiffnessOnVerticesFast mode.");  // otherwise we get nans
            springs[i].position_A->set(find_vertex_coords(m_attachment_vertices[i].rod_idx_A, m_attachment_vertices[i].vertex_A));
            springs[i].position_B->set(find_vertex_coords(m_attachment_vertices[i].rod_idx_B, m_attachment_vertices[i].vertex_B));
        }
    }
    else if (rest_vars_type == RestVarsType::Stiffness || rest_vars_type == RestVarsType::SpringAnchors || rest_vars_type == RestVarsType::StiffnessAndSpringAnchors){
        for (size_t i = 0; i < springs.size(); ++i) {
            if (spring_anchors.size() != springs.size())
                throw std::runtime_error("Cannot set sliding spring endpoints if spring anchors are not set");

            for (size_t ai = 0; ai < 2; ++ai) {
                size_t ri = spring_anchor_vars[i].getRodIdx(ai);
                std::shared_ptr<std::vector<Vec3_T>> points = std::make_shared<std::vector<Vec3_T>>(get_deformed_points_for_rod(ri));
                std::shared_ptr<std::vector<Real_>> rest_mat_vars = std::make_shared<std::vector<Real_>>(get_rest_material_vars_for_rod(ri));
                spring_anchors[i][ai]->set_trajectory(points);
                spring_anchors[i][ai]->set_rest_material_vars(rest_mat_vars);
                spring_anchors[i][ai]->set(spring_anchor_vars[i][ai]);
            }

            springs[i].position_A = spring_anchors[i][0];
            springs[i].position_B = spring_anchors[i][1];
        }
    }
    else
        throw std::runtime_error("Unknown rest vars type");
    BENCHMARK_STOP_TIMER_SECTION("Tencer update_spring_endpoint_coords");
}

template<typename Real_>
void Tencer_T<Real_>::initialize_spring_anchors(){
    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast){
        throw std::runtime_error("Cannot initialize sliding spring endpoints if rest vars type is StiffnessOnVerticesFast");
    }
    else if (rest_vars_type == RestVarsType::Stiffness || rest_vars_type == RestVarsType::SpringAnchors || rest_vars_type == RestVarsType::StiffnessAndSpringAnchors){
        spring_anchors.resize(spring_anchor_vars.size());
        for (size_t i = 0; i < spring_anchor_vars.size(); ++i){

            SpringAnchors anchors;
            for (size_t ai = 0; ai < 2; ++ai) {
                size_t ri = spring_anchor_vars[i].getRodIdx(ai);
                std::shared_ptr<std::vector<Vec3_T>> points = std::make_shared<std::vector<Vec3_T>>(get_deformed_points_for_rod(ri));
                std::shared_ptr<std::vector<Real_>> rest_mat_vars = std::make_shared<std::vector<Real_>>(get_rest_material_vars_for_rod(ri));
                if (ri < open_rods.size())
                    anchors[ai] = std::make_shared<LinearOpenSlidingNode_T<Real_>>(points, rest_mat_vars, spring_anchor_vars[i][ai]);
                else if (ri >= open_rods.size() && ri < open_rods.size() + closed_rods.size())
                    anchors[ai] = std::make_shared<LinearClosedSlidingNode_T<Real_>>(points, rest_mat_vars, spring_anchor_vars[i][ai]);
                else
                    throw std::runtime_error("initialize_spring_anchors: Rod index out of bounds");
            }
            spring_anchors[i] = anchors;
        }
    }
    else
        throw std::runtime_error("Unknown rest vars type");
}

template<typename Real_>
void Tencer_T<Real_>::initialize_rest_material_vars(){
    for (size_t ri = 0; ri < open_rods.size(); ++ri)
        rest_mat_vars_open_rods.push_back(compute_rest_material_vars_open_rod(open_rods[ri]));
    for (size_t ri = 0; ri < closed_rods.size(); ++ri)
        rest_mat_vars_closed_rods.push_back(compute_rest_material_vars_closed_rod(closed_rods[ri]));
}

template<typename Real_>
std::vector<Real_> Tencer_T<Real_>::compute_rest_material_vars_open_rod(const ElasticRod_T<Real_> rod) {
    const size_t nv = rod.numVertices();
    const std::vector<Real_> &restLengths = rod.restLengths();
    std::vector<Real_> restMatVars(nv);
    restMatVars[0] = 0;
    for (size_t i = 0; i < nv-1; i++)
        restMatVars[i+1] = restMatVars[i] + restLengths[i];
    return restMatVars;
}

template<typename Real_>
std::vector<Real_> Tencer_T<Real_>::compute_rest_material_vars_closed_rod(const PeriodicRod_T<Real_> rod) {
    return compute_rest_material_vars_open_rod(rod.rod);  // use the underlying ElasticRod (account for the overlapping edges) 
}

template<typename Real_>
void Tencer_T<Real_>::set_spring_anchors(const std::vector<SpringAnchorVars> &sav) {
    BENCHMARK_START_TIMER_SECTION("Tencer set_spring_anchors (vector)");
    if (rest_vars_type != RestVarsType::Stiffness && rest_vars_type != RestVarsType::SpringAnchors && rest_vars_type != RestVarsType::StiffnessAndSpringAnchors)
        throw std::runtime_error("Cannot set spring anchors.");
    spring_anchor_vars = sav;
    for (size_t i = 0; i < spring_anchor_vars.size(); ++i) {
        spring_anchors[i][0]->set(spring_anchor_vars[i][0]);
        spring_anchors[i][1]->set(spring_anchor_vars[i][1]);
    }
    BENCHMARK_STOP_TIMER_SECTION("Tencer set_spring_anchors (vector)");
}

template<typename Real_>
void Tencer_T<Real_>::set_spring_anchors(const Eigen::Ref<const VecX_T> &vars) {
    BENCHMARK_START_TIMER_SECTION("Tencer set_spring_anchors (Eigen)");
    if ((size_t)vars.size() != 2*springs.size())
        throw std::runtime_error("Wrong number of spring anchor variables");
    std::vector<SpringAnchorVars> sav(springs.size());
    for (size_t i = 0; i < springs.size(); ++i)
        sav[i] = SpringAnchorVars(spring_anchor_vars[i].rod_idx_A, vars[2*i], spring_anchor_vars[i].rod_idx_B, vars[2*i+1]);  // assume the supporting rods did not change
    set_spring_anchors(sav);
    BENCHMARK_STOP_TIMER_SECTION("Tencer set_spring_anchors (Eigen)");
}

template<typename Real_>
void Tencer_T<Real_>::set_rest_vars_type(RestVarsType rvt, bool updateState) {
    // Whenever switching to/from spring anchors, we need to update the corresponding variables (which might not have been initialized yet)
    auto old_rvt = rest_vars_type;
    rest_vars_type = rvt;
    for (auto &s : springs)
        s.set_rest_vars_type(rvt, updateState);

    if (updateState && old_rvt != rvt) {
        if (rvt == RestVarsType::StiffnessOnVerticesFast) {
            if (spring_anchor_vars.size() != springs.size())
                throw std::runtime_error("Cannot set rest_vars_type to StiffnessOnVerticesFast if spring anchors are not set.");
            // Round to closest node
            for (size_t i = 0; i < springs.size(); ++i) {
                const int rod_idx_A = spring_anchor_vars[i].rod_idx_A;
                const int rod_idx_B = spring_anchor_vars[i].rod_idx_B;
                const int vertex_A = spring_anchors[i][0]->closestNodeIndex();
                const int vertex_B = spring_anchors[i][1]->closestNodeIndex();
                m_attachment_vertices.push_back(SpringAttachmentVertices(rod_idx_A, vertex_A, rod_idx_B, vertex_B));
            }
            spring_anchor_vars.clear();
            spring_anchors.clear();
        }
        else if (rvt == RestVarsType::Stiffness || rvt == RestVarsType::SpringAnchors || rvt == RestVarsType::StiffnessAndSpringAnchors) {
            if (old_rvt == RestVarsType::StiffnessOnVerticesFast) {
                if (m_attachment_vertices.size() != springs.size())
                    throw std::runtime_error("Cannot set rest_vars_type to Stiffness, SpringAnchors or StiffnessAndSpringAnchors if previous rest_vars_type was StiffnessOnVerticesFast and attachment vertices are not set.");
                spring_anchor_vars = {};
                auto anchor_perturb = 0.0; // 1e-3 * rod.totalRestLength() / rod.numEdges();  // perturb initial anchor positions to avoid potentital singularities located exactly at rod's vertices
                for (size_t i = 0; i < springs.size(); ++i) {
                    const size_t rod_idx_A = m_attachment_vertices[i].rod_idx_A;
                    const size_t rod_idx_B = m_attachment_vertices[i].rod_idx_B;
                    const Real_ mat_coord_A = get_rest_material_vars_for_rod(rod_idx_A)[m_attachment_vertices[i].vertex_A] + anchor_perturb;
                    const Real_ mat_coord_B = get_rest_material_vars_for_rod(rod_idx_B)[m_attachment_vertices[i].vertex_B] + anchor_perturb;
                    spring_anchor_vars.push_back(SpringAnchorVars(rod_idx_A, mat_coord_A, rod_idx_B, mat_coord_B));
                }
                initialize_spring_anchors();
                set_spring_anchors(spring_anchor_vars);
                m_attachment_vertices.clear();
            }
            if (old_rvt == RestVarsType::Stiffness || old_rvt == RestVarsType::SpringAnchors || old_rvt == RestVarsType::StiffnessAndSpringAnchors) {
                // nothing to do, spring anchors are already set
            }
        }
        update_spring_endpoint_coords();
    }
}

template <typename Real_>
std::vector<Real> Tencer_T<Real_>::get_rest_vars_upper_bounds() const {
    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast)
        return std::vector<Real>(numRestVars(), std::numeric_limits<Real>::infinity());
    else if (rest_vars_type == RestVarsType::Stiffness)
        return std::vector<Real>(numRestVars(), std::numeric_limits<Real>::infinity());
    else if (rest_vars_type == RestVarsType::SpringAnchors)
        return get_spring_anchors_upper_bounds();
    else if (rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
        std::vector<Real> bounds(numRestVars(), std::numeric_limits<Real>::infinity());
        std::vector<Real> springAnchorsUpperBounds = get_spring_anchors_upper_bounds();
        size_t ns = springs.size();
        for (size_t i = ns; i < 3*ns; ++i)
            bounds[i] = springAnchorsUpperBounds[i - ns];
        return bounds;
    }
    else
        throw std::runtime_error("Unknown rest vars type");
}

template <typename Real_>
std::vector<Real> Tencer_T<Real_>::get_rest_vars_lower_bounds() const {
    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast)
        return std::vector<Real>(numRestVars(), 0);
    else if (rest_vars_type == RestVarsType::Stiffness)
        return std::vector<Real>(numRestVars(), 0);
    else if (rest_vars_type == RestVarsType::SpringAnchors)
        return get_spring_anchors_lower_bounds();
    else if (rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
        std::vector<Real> bounds(numRestVars(), 0);
        size_t ns = springs.size();
        std::vector<Real> springAnchorsLowerBounds = get_spring_anchors_lower_bounds();
        for (size_t i = ns; i < 3*ns; ++i)
            bounds[i] = springAnchorsLowerBounds[i - ns];
        return bounds;
    }
    else
        throw std::runtime_error("Unknown rest vars type");
}

template <typename Real_>
std::vector<Real> Tencer_T<Real_>::get_spring_anchors_upper_bounds() const {
    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast || rest_vars_type == RestVarsType::Stiffness)
        throw std::runtime_error("Requested spring anchors upper bounds for rest vars type StiffnessOnVerticesFast or Stiffness");
    else if (rest_vars_type == RestVarsType::SpringAnchors || rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
        std::vector<Real> bounds(2*springs.size(), std::numeric_limits<Real>::infinity());
        for (size_t si = 0; si < springs.size(); ++si) {
            for (size_t ai = 0; ai < 2; ++ai) {
                size_t rod_idx = spring_anchor_vars[si].getRodIdx(ai);
                if (rod_idx >= open_rods.size() && rod_idx < open_rods.size() + closed_rods.size()) // the rod is closed
                    continue;  // the periodicity is handled by the sliding node itself, no need to impose bounds 
                else if (rod_idx < open_rods.size()) // the rod is open
                    bounds[2*si + ai] = stripAutoDiff(open_rods[rod_idx].restLength());
            }
        }
        return bounds;
    }
    else
        throw std::runtime_error("Unknown rest vars type");
}

template <typename Real_>
std::vector<Real> Tencer_T<Real_>::get_spring_anchors_lower_bounds() const {
    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast || rest_vars_type == RestVarsType::Stiffness)
        throw std::runtime_error("Requested spring anchors lower bounds for rest vars type StiffnessOnVerticesFast or Stiffness");
    else if (rest_vars_type == RestVarsType::SpringAnchors || rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
        std::vector<Real> bounds(2*springs.size(), -std::numeric_limits<Real>::infinity());
        for (size_t si = 0; si < springs.size(); ++si) {
            for (size_t ai = 0; ai < 2; ++ai) {
                size_t rod_idx = spring_anchor_vars[si].getRodIdx(ai);
                if (rod_idx >= open_rods.size() && rod_idx < open_rods.size() + closed_rods.size()) // the rod is closed
                    continue;  // the periodicity is handled by the sliding node itself, no need to impose bounds 
                else if (rod_idx < open_rods.size()) // the rod is open
                    bounds[2*si + ai] = 0;
            }
        }
        return bounds;
    }
    else
        throw std::runtime_error("Unknown rest vars type");
}

template<typename Real_>
VecX_T<Real_> Tencer_T<Real_>::springs_coord_diff() const{
    VecX_T c = VecX_T(num_springs()*3);
    for (size_t i=0; i<springs.size(); ++i){
        c.segment(3*i,3) = springs[i].coords_diff();
    }
    return c;
}

template <typename Real_>
const std::vector<Real_> &Tencer_T<Real_>::get_rest_material_vars_for_rod(size_t rod_idx) const {
    if (rod_idx < open_rods.size())
        return rest_mat_vars_open_rods[rod_idx];
    else if (rod_idx < open_rods.size() + closed_rods.size())
        return rest_mat_vars_closed_rods[rod_idx - open_rods.size()];
    else
        throw std::runtime_error("Rod index out of bounds");
}

template <typename Real_>
const std::vector<Pt3_T<Real_>> Tencer_T<Real_>::get_deformed_points_for_rod(size_t rod_idx) const {
    if (rod_idx < open_rods.size())
        return open_rods[rod_idx].deformedPoints();
    else if (rod_idx < open_rods.size() + closed_rods.size())
        return closed_rods[rod_idx - open_rods.size()].rod.deformedPoints();  // fetching the underlying ElasticRod
    else
        throw std::runtime_error("Rod index out of bounds");
}


/*
 * Variable getters / setters
 */

template <typename Real_>
size_t Tencer_T<Real_>::numRestVars() const {
    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast || rest_vars_type == RestVarsType::Stiffness)
        return springs.size();
    else if (rest_vars_type == RestVarsType::SpringAnchors)
        return 2*springs.size();
    else if (rest_vars_type == RestVarsType::StiffnessAndSpringAnchors)
        return 3*springs.size();
    else
        throw std::runtime_error("Unknown rest vars type");
}

template <typename Real_>
size_t Tencer_T<Real_>::numVars(VariableMask vmask) const {
    switch (vmask)
    {
    case VariableMask::Defo:
        return numDefoVars();
    case VariableMask::Rest:
        return numRestVars();
    case VariableMask::All:
        return numDefoVars() + numRestVars();
    default:
        throw std::runtime_error("Unknown variable mask");
    }
}

template<typename Real_>
VecX_T<Real_> Tencer_T<Real_>::getDefoVars() const{
    VecX_T defoVars(numDefoVars());
    for (size_t i = 0; i < open_rods.size(); ++i){
        defoVars.segment(m_defoVarsIndexPerRod[i], open_rods[i].numDoF()) = open_rods[i].getDoFs();
    }
    for (size_t i = 0; i < closed_rods.size(); ++i){
        defoVars.segment(m_defoVarsIndexPerRod[open_rods.size() + i], closed_rods[i].numDoF()) = closed_rods[i].getDoFs();
    }
    return defoVars;
}

template <typename Real_>
VecX_T<Real_> Tencer_T<Real_>::getVars(VariableMask vmask) const {
    switch (vmask)
    {
    case VariableMask::Defo:
        return getDefoVars();
    case VariableMask::Rest:
        return getRestVars();
    case VariableMask::All: {
        VecX_T defoVars(numDefoVars() + numRestVars());
        defoVars.head(numDefoVars()) = getDefoVars();
        defoVars.tail(numRestVars()) = getRestVars();
        return defoVars;
    }
    default:
        throw std::runtime_error("Unknown variable mask");
    }
}

template<typename Real_>
void Tencer_T<Real_>::m_setDefoVars(const Eigen::Ref<const VecX_T> &vars){
    BENCHMARK_START_TIMER_SECTION("Tencer m_setDefoVars");
    assert((size_t)vars.size() == numDefoVars());
    for (size_t i = 0; i < open_rods.size(); ++i){
        open_rods[i].setDoFs(vars.segment(m_defoVarsIndexPerRod[i], open_rods[i].numDoF()));
    }
    for (size_t i = 0; i < closed_rods.size(); ++i){
        closed_rods[i].setDoFs(vars.segment(m_defoVarsIndexPerRod[open_rods.size() + i], closed_rods[i].numDoF()));
    }
    update_spring_endpoint_coords();
    BENCHMARK_STOP_TIMER_SECTION("Tencer m_setDefoVars");
}

template <typename Real_>
void Tencer_T<Real_>::setVars(const Eigen::Ref<const VecX_T> &vars, VariableMask vmask) {
    switch (vmask)
    {
    case VariableMask::Defo:
        setDefoVars(vars);
        break;
    case VariableMask::Rest:
        setRestVars(vars);
        break;
    case VariableMask::All:{
        setDefoVars(vars.head(numDefoVars()));
        setRestVars(vars.tail(numRestVars()));
        break;
    }
    default:
        throw std::runtime_error("Unknown variable mask");
    }
}



template<typename Real_>
VecX_T<Real_> Tencer_T<Real_>::getRestVars() const {
    VecX_T vars = VecX_T(numRestVars());
    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast || rest_vars_type == RestVarsType::Stiffness) {
        for (size_t i = 0; i < springs.size(); ++i) {
            vars[i] = springs[i].stiffness;
        }
    }
    else if (rest_vars_type == RestVarsType::SpringAnchors) {
        for (size_t i = 0; i < springs.size(); ++i) {
            vars[2*i]   = spring_anchors[i][0]->u();
            vars[2*i+1] = spring_anchors[i][1]->u();
        }
    }
    else if (rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
        const size_t ns = springs.size();
        for (size_t i = 0; i < ns; ++i) {
            vars[i] = springs[i].stiffness;
            vars[ns + 2*i]   = spring_anchors[i][0]->u();
            vars[ns + 2*i+1] = spring_anchors[i][1]->u();
        }
    }
    else
        throw std::runtime_error("Unknown rest vars type");
    return vars;
}

template<typename Real_>
void Tencer_T<Real_>::m_setRestVars(const Eigen::Ref<const VecX_T> &vars) {
    BENCHMARK_START_TIMER_SECTION("Tencer m_setRestVars");
    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast || rest_vars_type == RestVarsType::Stiffness) {
        for (size_t i = 0; i < springs.size(); ++i)
            springs[i].stiffness = vars[i];
    }
    else if (rest_vars_type == RestVarsType::SpringAnchors) {
        set_spring_anchors(vars);
        sparsityPatternUpdated = false;
    }
    else if (rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
        const size_t ns = springs.size();
        for (size_t i = 0; i < ns; ++i)
            springs[i].stiffness = vars[i];
        set_spring_anchors(vars.segment(ns, 2*ns));
        sparsityPatternUpdated = false;
    }
    else
        throw std::runtime_error("Unknown rest vars type");
    BENCHMARK_STOP_TIMER_SECTION("Tencer m_setRestVars");
}

template <typename Real_>
size_t Tencer_T<Real_>::kVarsOffset(RestVarsType rvt, VariableMask vmask) const {
    if (vmask == VariableMask::Defo)
        throw std::runtime_error("No kVarsOffset for VariableMask::Defo");
    else if (vmask == VariableMask::Rest) {
        if (rvt == RestVarsType::SpringAnchors)
            throw std::runtime_error("No kVarsOffset for RestVarsType::SpringAnchors");
        else /*if (rvt == RestVarsType::StiffnessOnVerticesFast || rest_vars_type == RestVarsType::Stiffness || rvt == RestVarsType::StiffnessAndSpringAnchors)*/
            return 0;
    }
    else /*if (vmask == VariableMask::All)*/ {
        if (rvt == RestVarsType::SpringAnchors)
            throw std::runtime_error("No kVarsOffset for RestVarsType::SpringAnchors");
        else /*if (rvt == RestVarsType::StiffnessOnVerticesFast || rest_vars_type == RestVarsType::Stiffness || rvt == RestVarsType::StiffnessAndSpringAnchors)*/
            return numDefoVars();
    }
}

template <typename Real_>
size_t Tencer_T<Real_>::uVarsOffset(RestVarsType rvt, VariableMask vmask) const {
    if (vmask == VariableMask::Defo)
        throw std::runtime_error("No uVarsOffset for VariableMask::Defo");
    else if (vmask == VariableMask::Rest) {
        if (rvt == RestVarsType::StiffnessOnVerticesFast || rvt == RestVarsType::Stiffness)
            throw std::runtime_error("No uVarsOffset for RestVarsType::StiffnessOnVerticesFast or Stiffness");
        else if (rvt == RestVarsType::SpringAnchors)
            return 0;
        else /*if (rvt == RestVarsType::StiffnessAndSpringAnchors)*/
            return springs.size();
    }
    else /*if (vmask == VariableMask::All)*/ {
        if (rvt == RestVarsType::StiffnessOnVerticesFast || rvt == RestVarsType::Stiffness)
            throw std::runtime_error("No uVarsOffset for RestVarsType::StiffnessOnVerticesFast or Stiffness");
        else if (rvt == RestVarsType::SpringAnchors)
            return numDefoVars();
        else /*if (rvt == RestVarsType::StiffnessAndSpringAnchors)*/
            return numDefoVars() + springs.size();
    }
}

/*
 * Energies, gradients, hessians
 */

template<typename Real_>
Real_ Tencer_T<Real_>::elastic_energy(EnergyType etype) const{
    Real_ energy = 0;
    for (const auto &r : open_rods)
        energy += r.energy(etype);
    for (const auto &r : closed_rods)
        energy += r.energy(etype);
    return energy;
}

template <typename Real_>
Real_ Tencer_T<Real_>::springs_energy() const{
    Real_ energy = 0;
    for (const auto spring : springs){
        energy += spring.energy();
    }
    return energy;
}


template <typename Real_>
Real_ Tencer_T<Real_>::energy(KnotEnergyType ke,EnergyType e) const{
    switch (ke){
    case KnotEnergyType::Elastic:
        return elastic_energy(e);
    case KnotEnergyType::Springs:
        return springs_energy();
    case KnotEnergyType::Full:
        return elastic_energy(e) + springs_energy();
    default:
        throw std::runtime_error("Not implemented for this knot energy type");
    }
}

template <typename Real_>
Real_ Tencer_T<Real_>::energy() const{
    return energy(KnotEnergyType::Full);
}


template <typename Real_>
VecX_T<Real_> Tencer_T<Real_>::gradient(KnotEnergyType ke, EnergyType e,  VariableMask vmask, bool updatedParametrization) const{
    switch (ke){
    case KnotEnergyType::Elastic:
        return rods_gradient(updatedParametrization,vmask,e);
    case KnotEnergyType::Springs:
        return springs_gradient(vmask);
    case KnotEnergyType::Full:
        return rods_gradient(updatedParametrization,vmask,e) + springs_gradient(vmask);
    default:
        throw std::runtime_error("Not implemented for this knot energy type");
    } 
}

template <typename Real_>
VecX_T<Real_> Tencer_T<Real_>::gradient(bool updatedParametrization, VariableMask vmask) const{
    return gradient(KnotEnergyType::Full, EnergyType::Full, vmask, updatedParametrization);
}


template <typename Real_>
VecX_T<Real_> Tencer_T<Real_>::rods_gradient(bool updatedParametrization, VariableMask vmask, EnergyType e) const{
    switch (vmask){
    case VariableMask::Defo:
        return rods_gradient_defo(updatedParametrization,e);
    case VariableMask::Rest:
        return VecX_T::Zero(numRestVars());
    case VariableMask::All:{
        VecX_T g = VecX_T::Zero(numDefoVars() + numRestVars());
        VecX_T g1 = rods_gradient_defo(updatedParametrization,e);
        g.head(numDefoVars()) = g1;
        return g;
    }
    default:
        throw std::runtime_error("Unknown variable mask");
    }
}

template <typename Real_>
VecX_T<Real_> Tencer_T<Real_>::springs_gradient(VariableMask vmask) const{
    switch (vmask){
    case VariableMask::Defo:
        return springs_gradient_defo();
    case VariableMask::Rest:
        return springs_gradient_rest();
    case VariableMask::All:{
        VecX_T g = VecX_T(numDefoVars() + numRestVars());
        VecX_T g1 = springs_gradient_defo();
        VecX_T g2 = springs_gradient_rest();
        g << g1, g2;
        return g;
    }
    default:
        throw std::runtime_error("Unknown variable mask");
    }
}

template <typename Real_>
VecX_T<Real_> Tencer_T<Real_>::rods_gradient_defo(bool updatedParametrization, EnergyType e) const{
    BENCHMARK_START_TIMER_SECTION("Tencer rods_gradient_defo");
    VecX_T result = VecX_T::Zero(numDefoVars());
    for (size_t i = 0; i < open_rods.size(); ++i) {
        result.segment(m_defoVarsIndexPerRod[i], open_rods[i].numDoF()) = open_rods[i].gradient(updatedParametrization,e,false,false);
    }
    for (size_t i = 0; i < closed_rods.size(); ++i) {
        result.segment(m_defoVarsIndexPerRod[open_rods.size() + i], closed_rods[i].numDoF()) = closed_rods[i].gradient(updatedParametrization,e);
    }
    BENCHMARK_STOP_TIMER_SECTION("Tencer rods_gradient_defo");
    return result;
}

template <typename Real_>
VecX_T<Real_> Tencer_T<Real_>::springs_gradient_defo() const{
    BENCHMARK_START_TIMER_SECTION("Tencer springs_gradient_defo");
    VecX_T result = VecX_T::Zero(numDefoVars());
    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast) {
        for (size_t i = 0; i < springs.size(); ++i){
            VecX_T part_g = springs[i].dE_dx();
            size_t idxA = m_defoVarsIndexPerRod[m_attachment_vertices[i].rod_idx_A] + m_attachment_vertices[i].vertex_A * 3;
            size_t idxB = m_defoVarsIndexPerRod[m_attachment_vertices[i].rod_idx_B] + m_attachment_vertices[i].vertex_B * 3;
            for (int j = 0; j < 3; ++j){
                result[idxA + j] += part_g[j];
                result[idxB + j] += part_g[j+3];
            }
        }
    }
    else if (rest_vars_type == RestVarsType::Stiffness || rest_vars_type == RestVarsType::SpringAnchors || rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
        for (size_t si = 0; si < springs.size(); ++si) {
            for (size_t ai = 0; ai < 2; ++ai) {
                const size_t rod_defo_offset = m_defoVarsIndexPerRod[spring_anchor_vars[si].getRodIdx(ai)];
                for (size_t ni : spring_anchors[si][ai]->stencilIndices())
                    result.template segment<3>(rod_defo_offset + 3*ni) += spring_anchors[si][ai]->dp_dx(ni) * springs[si].dE_dx(ai);
            }
        }
    }
    BENCHMARK_STOP_TIMER_SECTION("Tencer springs_gradient_defo");
    return result;
}

template <typename Real_>
VecX_T<Real_> Tencer_T<Real_>::springs_gradient_rest() const{
    BENCHMARK_START_TIMER_SECTION("Tencer springs_gradient_rest");
    VecX_T g = VecX_T(numRestVars());
    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast || rest_vars_type == RestVarsType::Stiffness) {
        for (long int i = 0; i < g.size(); ++i){
            g[i] = springs[i].dE_dk()[0];
        }
    }
    else if (rest_vars_type == RestVarsType::SpringAnchors) {
        for (size_t si = 0; si < springs.size(); ++si)
            g.template segment<2>(2*si) = springs[si].dE_du();
    }
    else if (rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
        const size_t ns = springs.size();
        for (size_t i = 0; i < ns; ++i) {
            g[i] = springs[i].dE_dk()[0];
            g.template segment<2>(ns + 2*i) = springs[i].dE_du();
        }
    }
    BENCHMARK_STOP_TIMER_SECTION("Tencer springs_gradient_rest");
    return g;
}


template <typename Real_>
void Tencer_T<Real_>::hessian(CSCMat_T &Hout, bool /*projectionMask*/, VariableMask vmask) const{
    return hessian(Hout, vmask, KnotEnergyType::Full, EnergyType::Full);
}


template <typename Real_>
void Tencer_T<Real_>::hessian(CSCMat_T &Hout, VariableMask vmask, KnotEnergyType ke, EnergyType e) const{
    BENCHMARK_SCOPED_TIMER_SECTION("hessian");
    Hout = hessianSparsityPattern(ke, vmask, 0.0);   // Hout should already have the correct sparsity pattern, this is just to ensure it does
    Hout.setZero();
    switch (ke){
    case KnotEnergyType::Elastic:
        rods_hessian(Hout,vmask,e);
        break;
    case KnotEnergyType::Springs:
        springs_hessian(Hout,vmask);
        break;
    case KnotEnergyType::Full:{
        rods_hessian(Hout,vmask,e);
        springs_hessian(Hout,vmask);
        break;
    }
    default:
        throw std::runtime_error("Not implemented for this knot energy type");
    }
}

template <typename Real_>
void Tencer_T<Real_>::rods_hessian(CSCMat_T &Hout, VariableMask vmask, EnergyType e) const {
    switch (vmask){
    case VariableMask::Defo:
        rods_hessian_defo(numDefoVars(), Hout,e);
        break;
    case VariableMask::Rest:
        // d^2Erod/dk^2 = 0
        break;
    case VariableMask::All:
        // d^2Erod/dk^2 = 0, dE/dk = 0
        rods_hessian_defo(numDefoVars() + numRestVars(), Hout,e);
        break;
    default:
        throw std::runtime_error("Unknown variable mask");
    }
}


template <typename Real_>
void Tencer_T<Real_>::springs_hessian(CSCMat_T &Hout, VariableMask vmask) const{
    switch (vmask){
    case VariableMask::Defo:
        springs_hessian_defo(Hout);
        break;
    case VariableMask::Rest:
        if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast || rest_vars_type == RestVarsType::Stiffness)
            break; // d2Espring/dk2 = 0
        else if (rest_vars_type == RestVarsType::SpringAnchors)
            springs_hessian_du2(Hout, VariableMask::Rest);
        else if (rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
            springs_hessian_dkdu(Hout, VariableMask::Rest);
            springs_hessian_du2(Hout, VariableMask::Rest);
        }
        break;
    case VariableMask::All:{
        springs_hessian_defo(Hout); //d2E/dx2
        if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast || rest_vars_type == RestVarsType::Stiffness)
            springs_hessian_dxdk(Hout); //d2E/dxdk
        else if (rest_vars_type == RestVarsType::SpringAnchors) {
            springs_hessian_dxdu(Hout); //d2E/dxdu
            springs_hessian_du2(Hout, VariableMask::All); //d2E/du2
        }
        else if (rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
            springs_hessian_dxdk(Hout); //d2E/dxdk
            springs_hessian_dxdu(Hout); //d2E/dxdu
            springs_hessian_dkdu(Hout, VariableMask::All); //d2E/dkdu
            springs_hessian_du2(Hout, VariableMask::All); //d2E/du2
        }
        break;
    }
    default:
        throw std::runtime_error("Unknown variable mask");
    }
}



template <typename Real_>
CSCMatrix<SuiteSparse_long, Real_> Tencer_T<Real_>::get_hessian(VariableMask vmask, KnotEnergyType ke, EnergyType e) const{
    CSCMat_T Hsp_csc = hessianSparsityPattern(ke,vmask,0.0);
    hessian(Hsp_csc,vmask,ke,e);
    return Hsp_csc;
}

template <typename Real_>
void Tencer_T<Real_>::rods_hessian_defo(size_t size, CSCMat_T &Hout, EnergyType e) const {
    CSCMat_T rod_hessian;
    for (size_t i = 0; i < num_rods(); ++i){
        if (i < open_rods.size()){
            auto &r = open_rods[i];
            rod_hessian = r.hessianSparsityPattern();
            r.hessian(rod_hessian, e, false);
        }
        else{
            auto &r = closed_rods[i - open_rods.size()];
            rod_hessian = r.hessianSparsityPattern();
            r.hessian(rod_hessian, e);
        }
        extendSparseMatrixSouthEast(rod_hessian, size - m_defoVarsIndexPerRod[i]);
        Hout.addWithDistinctSparsityPattern(rod_hessian, 1.0, m_defoVarsIndexPerRod[i], 0, std::numeric_limits<int>::max());
    }
}

template <typename Real_>
void Tencer_T<Real_>::springs_hessian_defo(CSCMat_T &Hout) const{
    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast) {
        for (size_t k = 0; k < springs.size(); ++k){
            MatX_T h = springs[k].d2E_dx2();
            size_t idxA = m_defoVarsIndexPerRod[m_attachment_vertices[k].rod_idx_A] + m_attachment_vertices[k].vertex_A * 3;
            size_t idxB = m_defoVarsIndexPerRod[m_attachment_vertices[k].rod_idx_B] + m_attachment_vertices[k].vertex_B * 3;
            for (int i = 0; i < 3; ++i){
                for (int j = i; j < 3; ++j){
                    Hout.addNZ(idxA+i,idxA+j,h(i,j));
                    Hout.addNZ(idxB+i,idxB+j,h(3+i,3+j));
                }
                for (int j = 0; j < 3; ++j){
                    Hout.addNZ(std::min(idxA+i, idxB+j), std::max(idxA+i, idxB+j), h(i,3+j));
                }
            }
        }
    }
    else if (rest_vars_type == RestVarsType::Stiffness || rest_vars_type == RestVarsType::SpringAnchors || rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
        auto set3x3Block = [](CSCMat_T &H, size_t a, size_t b, const Eigen::Matrix<Real_, 3, 3> &v) {
            size_t i = std::min(a, b);
            size_t j = std::max(a, b);
            for (size_t ii = i; ii < i+3; ii++) {
                for (size_t jj = j; jj < j+3; jj++) {
                    if (jj < ii) continue;
                    H.addNZ(ii, jj, v(ii-i, jj-j));
                }
            }
        };
        for (size_t si = 0; si < springs.size(); ++si) {
            // Diagonal blocks corresponding to each of the two sliding nodes separately
            // (size of the block depends on the stencil size, e.g. each block is 6x6 for linear sliding nodes)
            for (size_t ai = 0; ai < 2; ++ai) {
                const size_t rod_defo_offset = m_defoVarsIndexPerRod[spring_anchor_vars[si].getRodIdx(ai)];
                const auto &stencil_indices = spring_anchors[si][ai]->stencilIndices();
                for (size_t a : stencil_indices)
                    for (size_t b : stencil_indices)
                        if (b >= a)
                            set3x3Block(Hout, rod_defo_offset + 3*a, rod_defo_offset + 3*b, springs[si].d2E_dx2(ai, ai) * spring_anchors[si][ai]->dp_dx(a) * spring_anchors[si][ai]->dp_dx(b));
            }
            // Off-diagonal block corresponding to cross-interactions between the two sliding nodes 
            // (size of the block depends on the stencil size, e.g. it is 6x6 for linear sliding nodes)
            const auto &stencil_indices_first  = spring_anchors[si][0]->stencilIndices();
            const auto &stencil_indices_second = spring_anchors[si][1]->stencilIndices();
            const size_t rod_defo_offset_first  = m_defoVarsIndexPerRod[spring_anchor_vars[si].getRodIdx(0)];
            const size_t rod_defo_offset_second = m_defoVarsIndexPerRod[spring_anchor_vars[si].getRodIdx(1)];
            for (size_t a : stencil_indices_first)
                for (size_t b : stencil_indices_second)
                    set3x3Block(Hout, rod_defo_offset_first + 3*a, rod_defo_offset_second + 3*b, springs[si].d2E_dx2(0, 1) * spring_anchors[si][0]->dp_dx(a) * spring_anchors[si][1]->dp_dx(b));

            // Whenever the two sliding nodes are supported by the same rod and there is
            // an overlap between their stencils (i.e. short springs connecting close edges), 
            // we need to add a few extra terms
            if (spring_anchor_vars[si].getRodIdx(0) == spring_anchor_vars[si].getRodIdx(1)) {
                auto findCommonElements = [](const std::vector<size_t>& vec1, const std::vector<size_t>& vec2) {
                    std::unordered_set<size_t> unord_set(vec1.begin(), vec1.end());
                    std::vector<size_t> result;
                    for (const auto& elem : vec2)
                        if (unord_set.erase(elem))
                            result.push_back(elem);
                    return result;
                };
                const size_t rod_defo_offset = m_defoVarsIndexPerRod[spring_anchor_vars[si].getRodIdx(0)];  // guaranteed to be the same for both nodes
                const auto &stencil_overlap = findCommonElements(stencil_indices_first, stencil_indices_second);
                for (size_t a : stencil_overlap)
                    set3x3Block(Hout, rod_defo_offset + 3*a, rod_defo_offset + 3*a, springs[si].d2E_dx2(0, 1) * spring_anchors[si][0]->dp_dx(a) * spring_anchors[si][1]->dp_dx(a));   // all the other "additional" terms have already been added as usual, either as diagonal blocks or as off-diagonal blocks
            }
        }
    }
}

template <typename Real_>
void Tencer_T<Real_>::springs_hessian_dxdk(CSCMat_T &Hout) const{
    if (rest_vars_type != RestVarsType::StiffnessOnVerticesFast && rest_vars_type != RestVarsType::Stiffness && rest_vars_type != RestVarsType::StiffnessAndSpringAnchors)
        throw std::runtime_error("springs_hessian_dxdk called with rest_vars_type != StiffnessOnVerticeFast, Stiffness, or StiffnessAndSpringAnchors");

    int nv = numDefoVars();

    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast) {
        for (size_t k = 0; k < springs.size(); ++k){
            MatX_T h = springs[k].d2E_dxdk();
            size_t idxA = m_defoVarsIndexPerRod[m_attachment_vertices[k].rod_idx_A] + m_attachment_vertices[k].vertex_A * 3;
            size_t idxB = m_defoVarsIndexPerRod[m_attachment_vertices[k].rod_idx_B] + m_attachment_vertices[k].vertex_B * 3;
            for (int i = 0; i < 3; ++i){
                Hout.addNZ(idxA+i,nv+k,h(i));
                Hout.addNZ(idxB+i,nv+k,h(3+i));
            }
        }
    }
    else if (rest_vars_type == RestVarsType::StiffnessAndSpringAnchors || rest_vars_type == RestVarsType::Stiffness) {
        auto set3x1Block = [](CSCMat_T &H, size_t a, size_t b, Vec3_T v) {
            size_t i = std::min(a, b);
            size_t j = std::max(a, b);
            for (size_t k = 0; k < 3; k++)
                H.addNZ(i+k, j, v(k));
        };

        for (size_t si = 0; si < springs.size(); ++si) {
            for (size_t ai = 0; ai < 2; ++ai) {
                const size_t rod_defo_offset = m_defoVarsIndexPerRod[spring_anchor_vars[si].getRodIdx(ai)];
                for (size_t ni : spring_anchors[si][ai]->stencilIndices())
                    set3x1Block(Hout, rod_defo_offset + 3*ni, nv + si, springs[si].d2E_dxdk(ai) * spring_anchors[si][ai]->dp_dx(ni));
            }
        }
    }
}

template <typename Real_>
void Tencer_T<Real_>::springs_hessian_dxdu(CSCMat_T &Hout) const {
    if (rest_vars_type != RestVarsType::SpringAnchors && rest_vars_type != RestVarsType::StiffnessAndSpringAnchors)
        throw std::runtime_error("springs_hessian_dxdu called with rest_vars_type != SpringAnchors or StiffnessAndSpringAnchors");

    auto set3x1Block = [](CSCMat_T &H, size_t a, size_t b, Vec3_T v) {
        size_t i = std::min(a, b);
        size_t j = std::max(a, b);
        for (size_t k = 0; k < 3; k++)
            H.addNZ(i+k, j, v(k));
    };

    size_t u_offset = uVarsOffset(rest_vars_type, VariableMask::All);
    for (size_t si = 0; si < springs.size(); ++si) {
        for (size_t ai = 0; ai < 2; ++ai) {
            const size_t rod_defo_offset = m_defoVarsIndexPerRod[spring_anchor_vars[si].getRodIdx(ai)];
            for (size_t ni : spring_anchors[si][ai]->stencilIndices()) {
                for (size_t aj = 0; aj < 2; ++aj)
                    set3x1Block(Hout, rod_defo_offset + 3*ni, u_offset + 2*si+aj, springs[si].d2E_dxdu(ai, aj) * spring_anchors[si][ai]->dp_dx(ni));  // all the 3x1 blocks have the component d2E/dxdu * dp/dx
                set3x1Block(Hout, rod_defo_offset + 3*ni, u_offset + 2*si+ai, springs[si].dE_dx(ai) * spring_anchors[si][ai]->d2p_dxdu(ni));  // the 3x1 blocks with "mixed" node x(v) and material variable u have an additional component dE/dx * d2p/dxdu
            }
        }
    }
}

template <typename Real_>
void Tencer_T<Real_>::springs_hessian_du2(CSCMat_T &Hout, VariableMask vmask) const {
    if (rest_vars_type != RestVarsType::SpringAnchors && rest_vars_type != RestVarsType::StiffnessAndSpringAnchors)
        throw std::runtime_error("springs_hessian_du2 called with rest_vars_type != SpringAnchors or StiffnessAndSpringAnchors");

    if (vmask == VariableMask::Defo)
        throw std::runtime_error("springs_hessian_du2 not implemented for VariableMask::Defo");

    size_t u_offset = uVarsOffset(rest_vars_type, vmask);
    for (size_t si = 0; si < springs.size(); ++si) {
        Hout.addNZ(u_offset + 2*si  , u_offset + 2*si  , springs[si].d2E_du2(0, 0));
        Hout.addNZ(u_offset + 2*si+1, u_offset + 2*si+1, springs[si].d2E_du2(1, 1));
        Hout.addNZ(u_offset + 2*si  , u_offset + 2*si+1, springs[si].d2E_du2(0, 1));
    }
}

template <typename Real_>
void Tencer_T<Real_>::springs_hessian_dkdu(CSCMat_T &Hout, VariableMask vmask) const {
    if (rest_vars_type != RestVarsType::StiffnessAndSpringAnchors)
        throw std::runtime_error("springs_hessian_dkdu called with rest_vars_type != StiffnessAndSpringAnchors");

    if (vmask == VariableMask::Defo)
        throw std::runtime_error("springs_hessian_dkdu not implemented for VariableMask::Defo");

    size_t k_offset = kVarsOffset(rest_vars_type, vmask);
    size_t u_offset = uVarsOffset(rest_vars_type, vmask);
    for (size_t si = 0; si < springs.size(); ++si) {
        Hout.addNZ(k_offset + si, u_offset + 2*si  , springs[si].d2E_dkdu(0));
        Hout.addNZ(k_offset + si, u_offset + 2*si+1, springs[si].d2E_dkdu(1));
    }
}

template <typename Real_>
CSCMatrix<SuiteSparse_long, Real_> Tencer_T<Real_>::hessianSparsityPattern(Real_ val, VariableMask vmask) const{
    return hessianSparsityPattern(KnotEnergyType::Full,vmask,val);
}

template <typename Real_>
CSCMatrix<SuiteSparse_long, Real_> Tencer_T<Real_>::hessianSparsityPattern(KnotEnergyType ke,VariableMask vmask,Real_ val) const{
    switch (ke){
    case KnotEnergyType::Elastic:
        return rods_hessianSparsityPattern(val,vmask);
    case KnotEnergyType::Springs:
        return springs_hessianSparsityPattern(val,vmask);
    case KnotEnergyType::Full:{
        CSCMat_T res = rods_hessianSparsityPattern(val,vmask);
        res.addWithDistinctSparsityPattern(springs_hessianSparsityPattern(val,vmask));
        return res;
    }
    default:
        throw std::runtime_error("Not implemented for this knot energy type");
    }
}

template <typename Real_>
CSCMatrix<SuiteSparse_long, Real_> Tencer_T<Real_>::rods_hessianSparsityPattern(Real_ val, VariableMask vmask) const{
    switch (vmask){
    case VariableMask::Defo:
        return rods_hessianSparsityPattern_defo(val);
    case VariableMask::Rest: {
        size_t nv = numRestVars();
        TripletMatrix<Triplet<Real_>> Hsp(nv, nv);
        Hsp.symmetry_mode = TripletMatrix<Triplet<Real_>>::SymmetryMode::UPPER_TRIANGLE;
        return CSCMat_T(Hsp);
        }
    case VariableMask::All:{
        size_t nv = numDefoVars();
        size_t nd = numRestVars();
        CSCMat_T res = rods_hessianSparsityPattern_defo(val);
        // resize matrix
        extendSparseMatrixSouthEast(res,nv+nd);
        return res;
    }
    default:
        throw std::runtime_error("Unknown variable mask");
    }
}

template <typename Real_>
CSCMatrix<SuiteSparse_long, Real_> Tencer_T<Real_>::rods_hessianSparsityPattern_defo(Real_ val) const{
    CSCMat_T result(numDefoVars(), numDefoVars());
    CSCMat_T rod_sparsity;
    for (size_t i = 0; i < num_rods(); ++i){
        if (i < open_rods.size()){
            auto &r = open_rods[i];
            rod_sparsity = r.hessianSparsityPattern();
        }
        else {
            auto &r = closed_rods[i - open_rods.size()];
            rod_sparsity = r.hessianSparsityPattern();
        }
        extendSparseMatrixSouthEast(rod_sparsity, numDefoVars() - m_defoVarsIndexPerRod[i]); 
        result.addWithDistinctSparsityPattern(rod_sparsity, 1.0, m_defoVarsIndexPerRod[i], 0, std::numeric_limits<int>::max());
    }
    result.symmetry_mode = CSCMat_T::SymmetryMode::UPPER_TRIANGLE;
    result.fill(val);
    return result;
}

template <typename Real_>
CSCMatrix<SuiteSparse_long, Real_> Tencer_T<Real_>::springs_hessianSparsityPattern(Real_ val, VariableMask vmask) const{
    switch (vmask){
    case VariableMask::Defo:
        return springs_hessianSparsityPattern_defo(numDefoVars(), val);
    case VariableMask::Rest: {
        if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast || rest_vars_type == RestVarsType::Stiffness) {  // empty matrix
            TripletMatrix<Triplet<Real_>> Hsp(numRestVars(), numRestVars());
            Hsp.symmetry_mode = TripletMatrix<Triplet<Real_>>::SymmetryMode::UPPER_TRIANGLE;
            return CSCMat_T(Hsp);
        }
        else if (rest_vars_type == RestVarsType::SpringAnchors) 
            return springs_hessianSparsityPattern_du2(val, vmask);
        else if (rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
            CSCMat_T res = springs_hessianSparsityPattern_dkdu(val, vmask);
            res.addWithDistinctSparsityPattern(springs_hessianSparsityPattern_du2(val, vmask));
            return res;
        }
        else throw std::runtime_error("Unknown rest var type");
    }
    case VariableMask::All:{
        size_t nv = numDefoVars();
        size_t nd = numRestVars();
        CSCMat_T res = springs_hessianSparsityPattern_defo(nv+nd, val); //d2E/dx2
        if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast || rest_vars_type == RestVarsType::Stiffness)
            res.addWithDistinctSparsityPattern(springs_hessianSparsityPattern_dxdk(val)); //d2E/dxdk
        else if (rest_vars_type == RestVarsType::SpringAnchors) {
            res.addWithDistinctSparsityPattern(springs_hessianSparsityPattern_dxdu(val)); //d2E/dxdu
            res.addWithDistinctSparsityPattern(springs_hessianSparsityPattern_du2(val, vmask));
        }
        else if (rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
            res.addWithDistinctSparsityPattern(springs_hessianSparsityPattern_dxdk(val)); //d2E/dxdk
            res.addWithDistinctSparsityPattern(springs_hessianSparsityPattern_dxdu(val)); //d2E/dxdu
            res.addWithDistinctSparsityPattern(springs_hessianSparsityPattern_dkdu(val, vmask)); //d2E/dkdu
            res.addWithDistinctSparsityPattern(springs_hessianSparsityPattern_du2(val, vmask)); //d2E/du2
        }
        return res;
    }
    default:
        throw std::runtime_error("Unknown variable mask");
    }
}

template <typename Real_>
CSCMatrix<SuiteSparse_long, Real_> Tencer_T<Real_>::springs_hessianSparsityPattern_defo(size_t size, Real_ val) const{
    TripletMatrix<Triplet<Real_>> Hsp(size, size);
    Hsp.symmetry_mode = TripletMatrix<Triplet<Real_>>::SymmetryMode::UPPER_TRIANGLE;
    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast) {
        for (size_t k = 0; k < springs.size(); ++k){
            size_t idxA = m_defoVarsIndexPerRod[m_attachment_vertices[k].rod_idx_A] + m_attachment_vertices[k].vertex_A * 3;
            size_t idxB = m_defoVarsIndexPerRod[m_attachment_vertices[k].rod_idx_B] + m_attachment_vertices[k].vertex_B * 3;
            for (int i = 0; i < 3; ++i){
                for (int j = i; j < 3; ++j){
                    Hsp.addNZ(idxA+i,idxA+j,1.0);
                    Hsp.addNZ(idxB+i,idxB+j,1.0);
                }
                for (int j = 0; j < 3; ++j){
                    Hsp.addNZ(std::min(idxA+i, idxB+j), std::max(idxA+i, idxB+j), 1.0);
                }
            }
        }
    }
    else if (rest_vars_type == RestVarsType::Stiffness || rest_vars_type == RestVarsType::SpringAnchors || rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
        auto set3x3Block = [](TripletMatrix<Triplet<Real_>> &triplets, size_t a, size_t b) {
            size_t i = std::min(a, b);
            size_t j = std::max(a, b);
            for (size_t ii = i; ii < i+3; ii++) {
                for (size_t jj = j; jj < j+3; jj++) {
                    if (jj < ii) continue;
                    triplets.addNZ(ii, jj, 1.0);
                }
            }
        };
        for (size_t si = 0; si < springs.size(); ++si) {
            // Diagonal blocks corresponding to each of the two sliding nodes separately
            // (size of the block depends on the stencil size, e.g. each block is 6x6 for linear sliding nodes)
            for (size_t ai = 0; ai < 2; ++ai) {
                const size_t rod_defo_offset = m_defoVarsIndexPerRod[spring_anchor_vars[si].getRodIdx(ai)];
                const auto &stencil_indices = spring_anchors[si][ai]->stencilIndices();
                for (size_t a : stencil_indices)
                    for (size_t b : stencil_indices)
                        if (b >= a)
                            set3x3Block(Hsp, rod_defo_offset + 3*a, rod_defo_offset + 3*b);
            }
            // Off-diagonal block corresponding to cross-interactions between the two sliding nodes 
            // (size of the block depends on the stencil size, e.g. it is 6x6 for linear sliding nodes)
            const auto &stencil_indices_first  = spring_anchors[si][0]->stencilIndices();
            const auto &stencil_indices_second = spring_anchors[si][1]->stencilIndices();
            const size_t rod_defo_offset_first  = m_defoVarsIndexPerRod[spring_anchor_vars[si].getRodIdx(0)];
            const size_t rod_defo_offset_second = m_defoVarsIndexPerRod[spring_anchor_vars[si].getRodIdx(1)];
            for (size_t a : stencil_indices_first)
                for (size_t b : stencil_indices_second)
                    set3x3Block(Hsp, rod_defo_offset_first + 3*a, rod_defo_offset_second + 3*b);
        }
    }
    CSCMat_T Hsp_csc(Hsp);
    Hsp_csc.fill(val);
    return Hsp_csc;
}

template <typename Real_>
CSCMatrix<SuiteSparse_long, Real_> Tencer_T<Real_>::springs_hessianSparsityPattern_dxdk(Real_ val) const{
    if (rest_vars_type != RestVarsType::StiffnessOnVerticesFast && rest_vars_type != RestVarsType::Stiffness && rest_vars_type != RestVarsType::StiffnessAndSpringAnchors)
        throw std::runtime_error("springs_hessianSparsityPattern_dxdk called with rest_vars_type != StiffnessOnVerticesFast, Stiffness or StiffnessAndSpringAnchors");

    size_t nv = numDefoVars();
    size_t nd = numRestVars();
    TripletMatrix<Triplet<Real_>> Hout(nv+nd, nv+nd);
    Hout.symmetry_mode = TripletMatrix<Triplet<Real_>>::SymmetryMode::UPPER_TRIANGLE;

    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast) {
        for (size_t k = 0; k < springs.size(); ++k){
            size_t idxA = m_defoVarsIndexPerRod[m_attachment_vertices[k].rod_idx_A] + m_attachment_vertices[k].vertex_A * 3;
            size_t idxB = m_defoVarsIndexPerRod[m_attachment_vertices[k].rod_idx_B] + m_attachment_vertices[k].vertex_B * 3;
            for (int i = 0; i < 3; ++i){
                Hout.addNZ(idxA+i,nv+k, 1.0);
                Hout.addNZ(idxB+i,nv+k, 1.0);
            }
        }
    }
    else if (rest_vars_type == RestVarsType::StiffnessAndSpringAnchors || rest_vars_type == RestVarsType::Stiffness) {
        auto set3x1Block = [](TripletMatrix<Triplet<Real_>> &triplets, size_t a, size_t b) {
            size_t i = std::min(a, b);
            size_t j = std::max(a, b);
            for (size_t k = i; k < i+3; k++)
                triplets.addNZ(k, j, 1.0);
        };

        for (size_t si = 0; si < springs.size(); ++si) {
            for (size_t ai = 0; ai < 2; ++ai) {
                const size_t rod_defo_offset = m_defoVarsIndexPerRod[spring_anchor_vars[si].getRodIdx(ai)];
                for (size_t ni : spring_anchors[si][ai]->stencilIndices())
                    set3x1Block(Hout, rod_defo_offset + 3*ni, nv + si);
            }
        }
    }
    CSCMat_T H(Hout);
    H.fill(val);
    return H;
}

template <typename Real_>
CSCMatrix<SuiteSparse_long, Real_> Tencer_T<Real_>::springs_hessianSparsityPattern_dxdu(Real_ val) const {
    if (rest_vars_type != RestVarsType::SpringAnchors && rest_vars_type != RestVarsType::StiffnessAndSpringAnchors)
        throw std::runtime_error("springs_hessianSparsityPattern_dxdu called with rest_vars_type != SpringAnchors or StiffnessAndSpringAnchors");
    auto set3x1Block = [](TripletMatrix<Triplet<Real_>> &triplets, size_t a, size_t b) {
        size_t i = std::min(a, b);
        size_t j = std::max(a, b);
        for (size_t k = i; k < i+3; k++)
            triplets.addNZ(k, j, 1.0);
    };
    size_t nv = numDefoVars();
    size_t nd = numRestVars();
    size_t u_offset = uVarsOffset(rest_vars_type, VariableMask::All);

    TripletMatrix<Triplet<Real_>> Hout(nv+nd, nv+nd);
    Hout.symmetry_mode = TripletMatrix<Triplet<Real_>>::SymmetryMode::UPPER_TRIANGLE;
    for (size_t si = 0; si < springs.size(); ++si) {
        for (size_t ai = 0; ai < 2; ++ai) {
            const size_t rod_defo_offset = m_defoVarsIndexPerRod[spring_anchor_vars[si].getRodIdx(ai)];
            for (size_t ni : spring_anchors[si][ai]->stencilIndices())
                for (size_t aj = 0; aj < 2; ++aj)
                    set3x1Block(Hout, rod_defo_offset + 3*ni, u_offset + 2*si+aj);
        }
    }
    CSCMat_T H(Hout);
    H.fill(val);
    return H;
}

template <typename Real_>
CSCMatrix<SuiteSparse_long, Real_> Tencer_T<Real_>::springs_hessianSparsityPattern_du2(Real_ val, VariableMask vmask) const {
    if (rest_vars_type != RestVarsType::SpringAnchors && rest_vars_type != RestVarsType::StiffnessAndSpringAnchors)
        throw std::runtime_error("springs_hessianSparsityPattern_du2 called with rest_vars_type != SpringAnchors or StiffnessAndSpringAnchors");

    if (vmask == VariableMask::Defo)
        throw std::runtime_error("Calling springs_hessianSparsityPattern_du2 with VariableMask::Defo is not supported");

    size_t u_offset = uVarsOffset(rest_vars_type, vmask);

    size_t hsp_size;
    if (vmask == VariableMask::Rest)
        hsp_size = numRestVars();
    else /*if (vmask == VariableMask::All)*/
        hsp_size = numDefoVars() + numRestVars();

    TripletMatrix<Triplet<Real_>> Htrip(hsp_size, hsp_size);
    Htrip.symmetry_mode = TripletMatrix<Triplet<Real_>>::SymmetryMode::UPPER_TRIANGLE;
    for (size_t si = 0; si < springs.size(); ++si) {
        Htrip.addNZ(u_offset + 2*si  , u_offset + 2*si  , 1.0);
        Htrip.addNZ(u_offset + 2*si+1, u_offset + 2*si+1, 1.0);
        Htrip.addNZ(u_offset + 2*si  , u_offset + 2*si+1, 1.0);
    }
    CSCMat_T H(Htrip);
    H.fill(val);
    return H;
}

template <typename Real_>
CSCMatrix<SuiteSparse_long, Real_> Tencer_T<Real_>::springs_hessianSparsityPattern_dkdu(Real_ val, VariableMask vmask) const {
    if (rest_vars_type != RestVarsType::StiffnessAndSpringAnchors)
        throw std::runtime_error("springs_hessianSparsityPattern_dkdu called with rest_vars_type != StiffnessAndSpringAnchors");

    size_t k_offset = kVarsOffset(rest_vars_type, vmask);
    size_t u_offset = uVarsOffset(rest_vars_type, vmask);

    size_t hsp_size = 0;
    if (vmask == VariableMask::Defo)
        throw std::runtime_error("Calling springs_hessianSparsityPattern_dkdu with VariableMask::Defo is not supported");
    else if (vmask == VariableMask::Rest)
        hsp_size = numRestVars();
    else if (vmask == VariableMask::All)
        hsp_size = numDefoVars() + numRestVars();
    else
        throw std::runtime_error("Unknown variable mask");

    TripletMatrix<Triplet<Real_>> Htrip(hsp_size, hsp_size);
    Htrip.symmetry_mode = TripletMatrix<Triplet<Real_>>::SymmetryMode::UPPER_TRIANGLE;
    for (size_t si = 0; si < springs.size(); ++si) {
        Htrip.addNZ(k_offset + si, u_offset + 2*si  , 1.0);
        Htrip.addNZ(k_offset + si, u_offset + 2*si+1, 1.0);
    }
    CSCMat_T H(Htrip);
    H.fill(val);
    return H;
}




/*
 * apply_hessian : used in inverse design optimization
 *
 * d2E/dx dxp v = (d2E/dx2 , d2E/dxdp) v
 *
 *                    IN   
 *               Defo  |  Rest
 *             ________________
 *      Defo  |        |       |
 * OUT         ________________
 *      Rest  |        |       |
 *             ________________
 */
template <typename Real_>
Eigen::Matrix<Real_, -1, 1> Tencer_T<Real_>::apply_hessian(const VecX_T &v, const HessianComputationMask &mask){
    int nx = numDefoVars();
    int np = numRestVars();
    VecX_T res = VecX_T::Zero(nx+np);
    if (mask.dof_in){
        if (mask.dof_out){
            apply_rods_hessian_defo_defo(v,res, mask);
            apply_springs_hessian_defo_defo(v,res);
        }
        if (mask.restlen_out){
            apply_springs_hessian_defo_rest(v,res);
        }
    }
    if (mask.restlen_in){
        if (mask.dof_out){
            apply_springs_hessian_rest_defo(v,res);
        }
        if (mask.restlen_out){
            apply_springs_hessian_rest_rest(v,res);
        }
    }
    return res;
}

template <typename Real_>
void Tencer_T<Real_>::apply_rods_hessian_defo_defo(const VecX_T &v, VecX_T &res,  const HessianComputationMask &mask){
    for (size_t k = 0; k < open_rods.size(); ++k){
        res.segment(m_defoVarsIndexPerRod[k], open_rods[k].numDoF()) += open_rods[k].applyHessian(v.segment(m_defoVarsIndexPerRod[k], open_rods[k].numDoF()),false, mask);
    }
    for (size_t k = 0; k < closed_rods.size(); ++k){
        res.segment(m_defoVarsIndexPerRod[open_rods.size() + k], closed_rods[k].numDoF()) += closed_rods[k].applyHessian(v.segment(m_defoVarsIndexPerRod[open_rods.size() + k], closed_rods[k].numDoF()), mask);
    }
}

template <typename Real_>
void Tencer_T<Real_>::apply_springs_hessian_defo_defo(const VecX_T &v, VecX_T &res){
    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast) {
        for (size_t k = 0; k < springs.size(); ++k){
            size_t idxA = m_defoVarsIndexPerRod[m_attachment_vertices[k].rod_idx_A] + m_attachment_vertices[k].vertex_A * 3;
            size_t idxB = m_defoVarsIndexPerRod[m_attachment_vertices[k].rod_idx_B] + m_attachment_vertices[k].vertex_B * 3;
            if (springs[k].rest_length > 1e-10){
                MatX_T h = springs[k].d2E_dx2();
                for (int i = 0; i < 3; ++i){
                    for (int j = 0; j < 3; ++j){
                        res[idxA + i] += h(i,j) * v[idxA+j];
                        res[idxB + i] += h(3+i,3+j) * v[idxB+j];
                        res[idxA+i] += h(i,3+j) * v[idxB+j];
                        res[idxB+j] += h(3+i,j) * v[idxA+i];
                    }
                }
            }
            // simplify for zero rest length springs
            else {
                Real_ s = springs[k].stiffness;
                for (int i = 0; i < 3; ++i){
                    res[idxA+i] += s * v[idxA+i];
                    res[idxB+i] += s * v[idxB+i];
                    res[idxA+i] += -s * v[idxB+i];
                    res[idxB+i] += -s * v[idxA+i];
                }
            }
        }
    }
    else if (rest_vars_type == RestVarsType::Stiffness || rest_vars_type == RestVarsType::SpringAnchors || rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
        for (size_t si = 0; si < springs.size(); ++si) {
            // Diagonal blocks corresponding to each of the two sliding nodes separately
            for (size_t ai = 0; ai < 2; ++ai) {
                const size_t rod_defo_offset = m_defoVarsIndexPerRod[spring_anchor_vars[si].getRodIdx(ai)];
                const auto &stencil_indices = spring_anchors[si][ai]->stencilIndices();
                for (size_t a : stencil_indices)
                    for (size_t b : stencil_indices)
                        res.template segment<3>(rod_defo_offset + 3*a) += springs[si].d2E_dx2(ai, ai) * spring_anchors[si][ai]->dp_dx(a) * spring_anchors[si][ai]->dp_dx(b) * v.template segment<3>(rod_defo_offset + 3*b);
            }
            // Off-diagonal block corresponding to cross-interactions between the two sliding nodes
            for (size_t ai = 0; ai < 2; ++ai) {
                const size_t rod_defo_offset_a = m_defoVarsIndexPerRod[spring_anchor_vars[si].getRodIdx(ai)];
                const size_t rod_defo_offset_b = m_defoVarsIndexPerRod[spring_anchor_vars[si].getRodIdx(1-ai)];
                for (size_t a : spring_anchors[si][ai]->stencilIndices())
                    for (size_t b : spring_anchors[si][1-ai]->stencilIndices())
                        res.template segment<3>(rod_defo_offset_a+3*a) += springs[si].d2E_dx2(ai, 1-ai) * spring_anchors[si][ai]->dp_dx(a) * spring_anchors[si][1-ai]->dp_dx(b) * v.template segment<3>(rod_defo_offset_b+3*b);
            }
            // // Alternative, more verbose way to add the cross-interaction terms
            // const size_t rod_defo_offset_first  = m_defoVarsIndexPerRod[spring_anchor_vars[si].getRodIdx(0)];
            // const size_t rod_defo_offset_second = m_defoVarsIndexPerRod[spring_anchor_vars[si].getRodIdx(1)];
            // const auto &stencil_indices_first  = spring_anchors[si][0]->stencilIndices();
            // const auto &stencil_indices_second = spring_anchors[si][1]->stencilIndices();
            // for (size_t a : stencil_indices_first)
            //     for (size_t b : stencil_indices_second)
            //         res.template segment<3>(rod_defo_offset_first + 3*a) += springs[si].d2E_dx2(0, 1) * spring_anchors[si][0]->dp_dx(a) * spring_anchors[si][1]->dp_dx(b) * v.template segment<3>(rod_defo_offset_second + 3*b);
            // for (size_t a : stencil_indices_second)
            //     for (size_t b : stencil_indices_first)
            //         res.template segment<3>(rod_defo_offset_second + 3*a) += springs[si].d2E_dx2(1, 0) * spring_anchors[si][1]->dp_dx(a) * spring_anchors[si][0]->dp_dx(b) * v.template segment<3>(rod_defo_offset_first + 3*b);
        }
    }
}

template <typename Real_>
void Tencer_T<Real_>::apply_springs_hessian_defo_rest(const VecX_T &v, VecX_T &res){
    int nx = numDefoVars();
    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast) {
        // k-x block (with anchors at vertices)
        for (size_t k = 0; k < springs.size(); ++k){
            size_t idxA = m_defoVarsIndexPerRod[m_attachment_vertices[k].rod_idx_A] + m_attachment_vertices[k].vertex_A * 3;
            size_t idxB = m_defoVarsIndexPerRod[m_attachment_vertices[k].rod_idx_B] + m_attachment_vertices[k].vertex_B * 3;
            VecX_T h = springs[k].d2E_dxdk();
            for (int i = 0; i < 3; ++i){
                res[nx+k] += h[i] * v[idxA+i];
                res[nx+k] += h[3+i] * v[idxB+i];
            }
        }
    }
    if (rest_vars_type == RestVarsType::SpringAnchors || rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
        // u-x block
        size_t u_offset = uVarsOffset(rest_vars_type, VariableMask::All);
        for (size_t si = 0; si < springs.size(); ++si) {
            for (size_t ai = 0; ai < 2; ++ai) {
                const size_t rod_defo_offset = m_defoVarsIndexPerRod[spring_anchor_vars[si].getRodIdx(ai)];
                for (size_t ni : spring_anchors[si][ai]->stencilIndices()) {
                    for (size_t aj = 0; aj < 2; ++aj)
                        res[u_offset + 2*si+aj] += springs[si].d2E_dxdu(ai, aj).dot(spring_anchors[si][ai]->dp_dx(ni) * v.template segment<3>(rod_defo_offset + 3*ni));
                    res[u_offset + 2*si+ai] += springs[si].dE_dx(ai).dot(spring_anchors[si][ai]->d2p_dxdu(ni) * v.template segment<3>(rod_defo_offset + 3*ni));
                }
            }
        }
    }
    if (rest_vars_type == RestVarsType::Stiffness || rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
        // k-x block (with anchors on edges)
        size_t k_offset = kVarsOffset(rest_vars_type, VariableMask::All);
        for (size_t si = 0; si < springs.size(); ++si) {
            for (size_t ai = 0; ai < 2; ++ai) {
                const size_t rod_defo_offset = m_defoVarsIndexPerRod[spring_anchor_vars[si].getRodIdx(ai)];
                for (size_t ni : spring_anchors[si][ai]->stencilIndices())
                    res[k_offset + si] += springs[si].d2E_dxdk(ai).dot(spring_anchors[si][ai]->dp_dx(ni) * v.template segment<3>(rod_defo_offset + 3*ni));
            }
        }
    }
}

template <typename Real_>
void Tencer_T<Real_>::apply_springs_hessian_rest_defo(const VecX_T &v, VecX_T &res){
    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast) {
        // x-k block (with anchors at vertices)
        size_t k_offset = kVarsOffset(rest_vars_type, VariableMask::All);
        for (size_t k = 0; k < springs.size(); ++k){
            size_t idxA = m_defoVarsIndexPerRod[m_attachment_vertices[k].rod_idx_A] + m_attachment_vertices[k].vertex_A * 3;
            size_t idxB = m_defoVarsIndexPerRod[m_attachment_vertices[k].rod_idx_B] + m_attachment_vertices[k].vertex_B * 3;
            VecX_T h = springs[k].d2E_dxdk();
            for (int i = 0; i < 3; ++i){
                res[idxA+i] += h[i] * v[k_offset + k];
                res[idxB+i] += h[3+i] * v[k_offset + k];
            }
        }
    }
    if (rest_vars_type == RestVarsType::SpringAnchors || rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
        // x-u block
        size_t u_offset = uVarsOffset(rest_vars_type, VariableMask::All);
        for (size_t si = 0; si < springs.size(); ++si) {
            for (size_t ai = 0; ai < 2; ++ai) {
                const size_t rod_defo_offset = m_defoVarsIndexPerRod[spring_anchor_vars[si].getRodIdx(ai)];
                for (size_t ni : spring_anchors[si][ai]->stencilIndices()) {
                    for (size_t aj = 0; aj < 2; ++aj)
                        res.template segment<3>(rod_defo_offset + 3*ni) += springs[si].d2E_dxdu(ai, aj) * spring_anchors[si][ai]->dp_dx(ni) * v[u_offset + 2*si+aj];
                    res.template segment<3>(rod_defo_offset + 3*ni) += springs[si].dE_dx(ai) * spring_anchors[si][ai]->d2p_dxdu(ni) * v[u_offset + 2*si+ai];
                }
            }
        }
    }
    if (rest_vars_type == RestVarsType::Stiffness || rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
        // k-x block (with anchors on edges)
        size_t k_offset = kVarsOffset(rest_vars_type, VariableMask::All);
        for (size_t si = 0; si < springs.size(); ++si) {
            for (size_t ai = 0; ai < 2; ++ai) {
                const size_t rod_defo_offset = m_defoVarsIndexPerRod[spring_anchor_vars[si].getRodIdx(ai)];
                for (size_t ni : spring_anchors[si][ai]->stencilIndices())
                    res.template segment<3>(rod_defo_offset + 3*ni) += springs[si].d2E_dxdk(ai) * spring_anchors[si][ai]->dp_dx(ni) * v[k_offset + si];
            }
        }
    }
}

template <typename Real_>
void Tencer_T<Real_>::apply_springs_hessian_rest_rest(const VecX_T &v, VecX_T &res){
    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast || rest_vars_type == RestVarsType::Stiffness) {
        return; // d2E/dk2 = 0
    }
    else if (rest_vars_type == RestVarsType::SpringAnchors || rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
        // u-u block
        size_t u_offset = uVarsOffset(rest_vars_type, VariableMask::All); // using VariableMask::All because apply_hessian always acts on all-vars-size vectors; HessianComputationMask takes care of setting the right mask
        for (size_t si = 0; si < springs.size(); ++si) {
            res[u_offset + 2*si  ] += springs[si].d2E_du2(0, 0) * v[u_offset + 2*si  ];
            res[u_offset + 2*si+1] += springs[si].d2E_du2(1, 1) * v[u_offset + 2*si+1];
            res[u_offset + 2*si  ] += springs[si].d2E_du2(0, 1) * v[u_offset + 2*si+1];
            res[u_offset + 2*si+1] += springs[si].d2E_du2(1, 0) * v[u_offset + 2*si  ];
        }
        if (rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
            size_t k_offset = kVarsOffset(rest_vars_type, VariableMask::All); // using VariableMask::All because apply_hessian always acts on all-vars-size vectors; HessianComputationMask takes care of setting the right mask
            // k-u block
            for (size_t si = 0; si < springs.size(); ++si) {
                res[k_offset + si] += springs[si].d2E_dkdu(0) * v[u_offset + 2*si];
                res[k_offset + si] += springs[si].d2E_dkdu(1) * v[u_offset + 2*si+1];
            }
            // u-k block
            for (size_t si = 0; si < springs.size(); ++si) {
                res[u_offset + 2*si  ] += springs[si].d2E_dkdu(0) * v[k_offset + si];
                res[u_offset + 2*si+1] += springs[si].d2E_dkdu(1) * v[k_offset + si];
            }
        }
    }
}


/****************************************************************************
 **********   METHODS NEEDED BY THE DESIGNOPTIMIZATION FRAMEWORK ************
 ****************************************************************************/

template <typename Real_>
Eigen::Matrix<Real_, -1, 1> Tencer_T<Real_>::apply_hessian_defoVarsIn_restVarsOut(const VecX_T &v){
    HessianComputationMask mask;
    mask.dof_out = false;
    mask.restlen_in = false;
    return apply_hessian(v,mask);
}

template <typename Real_>
Eigen::Matrix<Real_, -1, 1> Tencer_T<Real_>::apply_hessian_defoVarsOut_restVarsIn(const VecX_T &v){
    HessianComputationMask mask;
    mask.dof_in = false;
    mask.restlen_out = false;
    return apply_hessian(v,mask);
}

template <typename Real_>
Eigen::Matrix<Real_, -1, 1> Tencer_T<Real_>::apply_hessian_defoVars_restVarsOut(const VecX_T &v){
    HessianComputationMask mask_dxpdx;
    mask_dxpdx.restlen_in = false;
    return apply_hessian(v,mask_dxpdx);
}


template struct Tencer_T<Real>;
template struct Tencer_T<ADReal>;