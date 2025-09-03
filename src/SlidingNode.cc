#include "SlidingNode.hh"

// ------------------------------------------------------
//                   LinearSlidingNode_T
// ------------------------------------------------------

// Get the index of all the nodes involved in the definition of the curve along which this node slides
template <typename Real_>
std::vector<size_t> LinearSlidingNode_T<Real_>::stencilIndices() const { 
    size_t i = stencilIndex();
    return { i, nextNodeIndex(i) };
}

// Get the index of the closest node (in the material domain, i.e. along the curve)
template <typename Real_>
size_t LinearSlidingNode_T<Real_>::closestNodeIndex() const {
    size_t i = stencilIndex();
    return uRest(i+1) - m_u < m_u - uRest(i) ? nextNodeIndex(i) : i;
}

template <typename Real_>
void LinearSlidingNode_T<Real_>::m_compute_p() {
    size_t i = this->stencilIndex();
    m_p = G(i, i+1) * x(i) + H(i, i+1) * x(i+1);
}

template <typename Real_>
typename LinearSlidingNode_T<Real_>::Pt3 LinearSlidingNode_T<Real_>::dp_du() const {
    size_t i = stencilIndex();
    return dG(i, i+1) * x(i) + dH(i, i+1) * x(i+1);
}

template <typename Real_>
Real_ LinearSlidingNode_T<Real_>::dp_dx(size_t ni) const {
    size_t i = stencilIndex();
    if      (ni == i)                return G(i, i+1);
    else if (ni == nextNodeIndex(i)) return H(i, i+1);
    else throw std::runtime_error("LinearClosedSlidingNode: dp/dx_" + std::to_string(ni) + " out of current stencil { " + std::to_string(i) + ", " + std::to_string(nextNodeIndex(i)) + " }");
}

template <typename Real_>
Real_ LinearSlidingNode_T<Real_>::d2p_dxdu(size_t ni) const {
    size_t i = stencilIndex();
    if      (ni == i)                return dG(i, i+1);
    else if (ni == nextNodeIndex(i)) return dH(i, i+1);
    else throw std::runtime_error("LinearClosedSlidingNode: dp/dx_" + std::to_string(ni) + " out of current stencil { " + std::to_string(i) + ", " + std::to_string(nextNodeIndex(i)) + " }");
}


// ------------------------------------------------------
//                LinearClosedSlidingNode_T
// ------------------------------------------------------

// Get the index of the supporting edge, where edge i connects node i to node i+1
template <typename Real_>
size_t LinearClosedSlidingNode_T<Real_>::stencilIndex() const { 
    return std::upper_bound(m_restMatVars->begin(), m_restMatVars->end(), m_u) - m_restMatVars->begin() - 1;  // u \in [u_i, u_{i+1})
}

// Set the sliding variable modulo L, L being the total rest length of the rod
template <typename Real_>
void LinearClosedSlidingNode_T<Real_>::m_set(Real_ u) {
    this->m_checkBounds(u); 
    Real_ unew = u;
    if      ((u > -L()) && (u <   0)) unew += L();
    else if ((u >  L()) && (u < 2*L())) unew -= L();

    m_u = unew;
    this->m_compute_p();
}

template <typename Real_>
void LinearClosedSlidingNode_T<Real_>::m_checkBounds(Real_ u) const { 
    if (u < -L() || u > 2*L())  // we set the sliding variable modulo L, but we only allow for "reasonable" values of u
        throw std::runtime_error("LinearClosedSlidingNode: u = " + std::to_string(stripAutoDiff(u)) + " out of bounds [" + std::to_string(-stripAutoDiff(L())) + ", " + std::to_string(2*stripAutoDiff(L())) + "]"); 
}


// ------------------------------------------------------
//                LinearOpenSlidingNode_T
// ------------------------------------------------------

// Get the index of the supporting edge, where edge i connects node i to node i+1
// NOTE: this is different from LinearClosedSlidingNode_T<Real_>::stencilIndex() 
//       because we need to consider the corner case represented by u = u_{nv-1} = L
template <typename Real_>
size_t LinearOpenSlidingNode_T<Real_>::stencilIndex() const { 
    size_t i = std::upper_bound(m_restMatVars->begin(), m_restMatVars->end(), m_u) - m_restMatVars->begin() - 1;  // u \in [u_i, u_{i+1})
    if (i == m_restMatVars->size() - 1) {
        assert((m_u - L()) < 1e-10);  // corner case: u = u_{nv-1} = L
        i--;  // we assign the node to the last edge nv-2
    }
    return i;
}

// Set the sliding variable
template <typename Real_>
void LinearOpenSlidingNode_T<Real_>::m_set(Real_ u) {
    this->m_checkBounds(u); 
    m_u = u;
    this->m_compute_p();
}

template <typename Real_>
void LinearOpenSlidingNode_T<Real_>::m_checkBounds(Real_ u) const { 
    if (u < 0 || u > L())
        throw std::runtime_error("LinearOpenSlidingNode: u = " + std::to_string(stripAutoDiff(u)) + " out of bounds [0, " + std::to_string(stripAutoDiff(L())) + "]"); 
}


template struct Node_T<Real>;
template struct Node_T<ADReal>;

template struct FixedNode_T<Real>;
template struct FixedNode_T<ADReal>;

template struct SlidingNode_T<Real>;
template struct SlidingNode_T<ADReal>;

template struct LinearSlidingNode_T<Real>;
template struct LinearSlidingNode_T<ADReal>;

template struct LinearClosedSlidingNode_T<Real>;
template struct LinearClosedSlidingNode_T<ADReal>;

template struct LinearOpenSlidingNode_T<Real>;
template struct LinearOpenSlidingNode_T<ADReal>;