#ifndef SLIDING_NODE_HH
#define SLIDING_NODE_HH

#include <MeshFEM/SparseMatrices.hh>
#include <ElasticRods/ElasticRod.hh>
#include <ElasticRods/PeriodicRod.hh>

template<typename Real_>
struct Node_T {
    using Pt3 = Pt3_T<Real_>;

    Node_T() { }

    Node_T(const Pt3 &p) : m_p(p) { }

    // Copy constructor, potentially converting from a different Real type
    template<typename Real_2>
    Node_T(const Node_T<Real_2> &n) : m_p(n.p()) {  }

    virtual ~Node_T() { }

    Pt3 p() const { return m_p; };
    void set(Pt3 p) { m_p = p; }

    virtual void set_trajectory(const std::shared_ptr<std::vector<Pt3>> &/*points*/) { throw std::runtime_error("set_trajectory not implemented for Node"); }
    virtual void set_rest_material_vars(const std::shared_ptr<std::vector<Real_>> &/*restMatVars*/) { throw std::runtime_error("set_rest_material_vars not implemented for Node"); }

    virtual void set(Real_ /*u*/) { throw std::runtime_error("set(u) not implemented for Node base type"); }
    virtual Pt3      dp_du()          const = 0;
    virtual Real_    dp_dx(size_t ni) const = 0;
    virtual Real_ d2p_dxdu(size_t ni) const = 0;

    virtual std::string type() const = 0;

protected:
    Pt3 m_p;
};


template<typename Real_>
struct FixedNode_T : public Node_T<Real_> {
    using Base = Node_T<Real_>;
    using Pt3 = typename Base::Pt3;
    using Base::p;
    using Base::m_p;
    using Base::set;

    FixedNode_T(const Pt3 &p) : Base(p) { }

    // Copy constructor, potentially converting from a different Real type
    template<typename Real_2>
    FixedNode_T(const FixedNode_T<Real_2> &n) : Base(n) {  }

    virtual ~FixedNode_T() { }

    virtual void set_trajectory(const std::shared_ptr<std::vector<Pt3>> &/*points*/) override { throw std::runtime_error("set_trajectory not implemented for FixedNode"); }
    virtual void set_rest_material_vars(const std::shared_ptr<std::vector<Real_>> &/*restMatVars*/) override { throw std::runtime_error("set_rest_material_vars not implemented for FixedNode"); }

    virtual void set(Real_ /*u*/)            override { throw std::runtime_error("set(u) not implemented for FixedNode"); }
    virtual Pt3      dp_du()          const override { return Pt3(0, 0, 0); }
    virtual Real_    dp_dx(size_t /*ni*/) const override { return 0; }
    virtual Real_ d2p_dxdu(size_t /*ni*/) const override { return 0; }

    virtual std::string type() const override { return "FixedNode"; }
};


template<typename Real_>
struct SlidingNode_T : public Node_T<Real_> {
    using Base = Node_T<Real_>;
    using Pt3 = typename Base::Pt3;
    using Base::p;
    using Base::dp_du;
    using Base::dp_dx;
    using Base::m_p;
    using Base::set;


    SlidingNode_T(std::shared_ptr<std::vector<Pt3>> points, std::shared_ptr<std::vector<Real_>> restMatVars, Real_ /*u*/ = 0.0)
        : m_trajectory(points), m_restMatVars(restMatVars) {  }

    // Copy constructor, potentially converting from a different Real type
    template<typename Real_2>
    SlidingNode_T(const SlidingNode_T<Real_2> &n) : Base(n), m_u(n.u()) {  }  // Note: we have not set the knot object yet, the caller will need to take care of that and of calling set(u) afterwards

    virtual ~SlidingNode_T() { }

    virtual void set(Pt3 /*p*/) final { throw std::runtime_error("set(Pt3) not allowed for SlidingNode, use set(Real) instead and provide the corresponding material variable"); }
    virtual void set(Real_ /*u*/) override { throw std::runtime_error("set(u) not implemented for SlidingNode base type"); }
    Real_ u() const { return m_u; }
    virtual void set_trajectory(const std::shared_ptr<std::vector<Pt3>> &points) override { m_trajectory = points; }
    virtual void set_rest_material_vars(const std::shared_ptr<std::vector<Real_>> &restMatVars) override { m_restMatVars = restMatVars; }

    // Get the index of the supporting edge, where edge i connects node i to node i+1
    virtual size_t stencilIndex() const = 0;

    // Get the index of all the nodes involved in the definition of the curve along which this node slides
    virtual std::vector<size_t> stencilIndices() const = 0;

    // Get the next node index
    virtual size_t nextNodeIndex(size_t i) const = 0;

    // Get the index of the closest node (in the material domain, i.e. along the curve)
    virtual size_t closestNodeIndex() const = 0;

    Real_ uRest(size_t i) const { return (*m_restMatVars)[i]; }
    virtual Pt3   x(size_t i) const = 0;
    virtual Real_ L()         const = 0;

    virtual std::string type() const override { return "SlidingNode"; }

protected:
    std::shared_ptr<std::vector<Pt3>> m_trajectory;
    std::shared_ptr<std::vector<Real_>> m_restMatVars;
    Real_ m_u;

    virtual void m_set(Real_ u) = 0;  // Set the sliding variable modulo L, L being the total rest length of the rod
    virtual void m_compute_p()  = 0; 

    virtual void m_checkBounds(Real_ u) const = 0;
};

template<typename Real_>
struct LinearSlidingNode_T : public SlidingNode_T<Real_> {
    using Base = SlidingNode_T<Real_>;
    using Base::m_trajectory;
    using Base::m_restMatVars;
    using Base::m_u;
    using Base::m_p;
    using Base::p;
    using Base::uRest;
    using typename Base::Pt3;

    LinearSlidingNode_T(std::shared_ptr<std::vector<Pt3>> points, std::shared_ptr<std::vector<Real_>> restMatVars, Real_ u = 0.0)
        : Base(points, restMatVars, u) { }

    // Copy constructor, potentially converting from a different Real type
    template<typename Real_2>
    LinearSlidingNode_T(const LinearSlidingNode_T<Real_2> &n) : Base(n) {  }

    virtual ~LinearSlidingNode_T() { }

    virtual void set(Real_ /*u*/) override { throw std::runtime_error("set(u) not implemented for LinearSlidingNode base type"); }

    virtual size_t stencilIndex() const override = 0;
    virtual std::vector<size_t> stencilIndices() const override;
    virtual size_t nextNodeIndex(size_t i) const override = 0;
    virtual size_t closestNodeIndex() const override;
    virtual Pt3   x(size_t i) const override = 0;
    virtual Real_ L()         const override = 0;

    virtual Pt3      dp_du()          const override;
    virtual Real_    dp_dx(size_t ni) const override;
    virtual Real_ d2p_dxdu(size_t ni) const override;

    virtual std::string type() const override { return "LinearSlidingNode"; }

protected:
    Real_  G(size_t i, size_t j) const { return (uRest(j) - m_u) * dH(i, j); }
    Real_  H(size_t i, size_t j) const { return (m_u - uRest(i)) * dH(i, j); }
    Real_ dG(size_t i, size_t j) const { return -dH(i, j); }
    Real_ dH(size_t i, size_t j) const { return 1 / (uRest(j) - uRest(i)); }

    virtual void m_set(Real_ u) override = 0;
    virtual void m_compute_p()  override;

    void m_checkBounds(Real_ u) const override = 0;
};

template<typename Real_>
struct LinearClosedSlidingNode_T : public LinearSlidingNode_T<Real_> {
    using Base = LinearSlidingNode_T<Real_>;
    using Base::m_trajectory;
    using Base::m_restMatVars;
    using Base::m_u;
    using Base::uRest;
    using typename Base::Pt3;

    LinearClosedSlidingNode_T(std::shared_ptr<std::vector<Pt3>> points, std::shared_ptr<std::vector<Real_>> restMatVars, Real_ u = 0.0)
        : Base(points, restMatVars, u) { m_set(u); }

    // Copy constructor, potentially converting from a different Real type
    template<typename Real_2>
    LinearClosedSlidingNode_T(const LinearClosedSlidingNode_T<Real_2> &n) : Base(n) {  }

    virtual ~LinearClosedSlidingNode_T() { }

    virtual void set(Real_ u) override { m_set(u); }

    virtual size_t stencilIndex() const override;
    virtual size_t nextNodeIndex(size_t i) const override { return cyclicIndexing(i+1); }

    // Note: in closed (periodic) rods, x is accessed by cyclic indexing, i.e. modulo the number of vertices.
    //       The rest material variables account for the two overlapping edges of the closed polyline, thus:
    //        - The total length is L = uRest(n-2), and not uRest(n-1);
    //        - uRest is *not* accessed cyclically, and 0 = uRest(0) ≠ uRest(n-2) = L.
    virtual Pt3   x(size_t i) const override { return (*m_trajectory)[cyclicIndexing(i)]; }
    virtual Real_ L()         const override { return uRest(m_restMatVars->size() - 2); }

    virtual std::string type() const override { return "LinearClosedSlidingNode"; }

protected:
    virtual void m_set(Real_ u) override;

    void m_checkBounds(Real_ u) const override;

    size_t numVerticesPeriodicTrajectory() const { return m_trajectory->size() - 2; }
    size_t cyclicIndexing(size_t i) const { return i % numVerticesPeriodicTrajectory(); }
};

template<typename Real_>
struct LinearOpenSlidingNode_T : public LinearSlidingNode_T<Real_> {
    using Base = LinearSlidingNode_T<Real_>;
    using Base::m_trajectory;
    using Base::m_restMatVars;
    using Base::m_u;
    using Base::uRest;
    using typename Base::Pt3;

    LinearOpenSlidingNode_T(std::shared_ptr<std::vector<Pt3>> points, std::shared_ptr<std::vector<Real_>> restMatVars, Real_ u = 0.0)
        : Base(points, restMatVars, u) { m_set(u); }

    // Copy constructor, potentially converting from a different Real type
    template<typename Real_2>
    LinearOpenSlidingNode_T(const LinearOpenSlidingNode_T<Real_2> &n) : Base(n) {  }

    virtual ~LinearOpenSlidingNode_T() { }

    virtual void set(Real_ u) override { m_set(u); }

    virtual size_t stencilIndex() const override;
    virtual size_t nextNodeIndex(size_t i) const override { return i < (m_trajectory->size() - 1) ? i + 1 : throw std::runtime_error("nextNodeIndex " + std::to_string(i) + " out of bounds"); }
    virtual Pt3   x(size_t i) const override { return (*m_trajectory)[i]; }
    virtual Real_ L()         const override { return uRest(m_restMatVars->size() - 1); }

    virtual std::string type() const override { return "LinearOpenSlidingNode"; }

protected:
    virtual void m_set(Real_ u) override;

    void m_checkBounds(Real_ u) const override;
};

#endif /* end of include guard: SLIDING_NODE_HH */