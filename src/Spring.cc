#include "Spring.hh"

template <typename Real_>
Spring_T<Real_>::Spring_T(Vec3_T &pA, Vec3_T &pB, Real_ s, double l) 
    : position_A(std::make_shared<FixedNode_T<Real_>>(pA)), 
      position_B(std::make_shared<FixedNode_T<Real_>>(pB)), 
      stiffness(s), rest_length(l) 
      {  }

template <typename Real_>
Spring_T<Real_>::Spring_T(const std::shared_ptr<Node_T<Real_>> &pA, const std::shared_ptr<Node_T<Real_>> &pB, Real_ s, double l) 
    : position_A(pA), position_B(pB), stiffness(s), rest_length(l) 
    {  }

template <typename Real_>
Real_ Spring_T<Real_>::energy() const{
    Vec3_T d = position_A->p() - position_B->p();
    Real_ dist = sqrt(d.dot(d));
    return 0.5 * stiffness * (dist - rest_length) * (dist - rest_length);
}


template<typename Real_>
typename Spring_T<Real_>::Vec3_T Spring_T<Real_>::dE_dx(size_t i) const {
    Vec3_T d = position_A->p() - position_B->p();
    Real_ dist = sqrt(d.dot(d));
    if      (i == 0) return   stiffness * (1 - rest_length/dist) * d;
    else if (i == 1) return - stiffness * (1 - rest_length/dist) * d;
    else throw std::runtime_error("dE_dx: index " + std::to_string(i) + " out of bounds");
}

template<typename Real_>
Real_ Spring_T<Real_>::dE_du(size_t i) const {
    if (rest_vars_type != RestVarsType::SpringAnchors && rest_vars_type != RestVarsType::StiffnessAndSpringAnchors)
        throw std::runtime_error("dE_du() called for a spring with rest_vars_type != RestVarsType::SpringAnchors or RestVarsType::StiffnessAndSpringAnchors");

    if      (i == 0) return dE_dx(0).dot(position_A->dp_du());
    else if (i == 1) return dE_dx(1).dot(position_B->dp_du());
    else throw std::runtime_error("dE_du: index " + std::to_string(i) + " out of bounds");
}

template <typename Real_>
typename Spring_T<Real_>::Mat3_T Spring_T<Real_>::d2E_dx2(size_t i, size_t j) const {
    if (rest_length == 0.0) {
        if      ((i == 0 && j == 0) || (i == 1 && j == 1)) return   stiffness * Mat3_T::Identity();
        else if ((i == 0 && j == 1) || (i == 1 && j == 0)) return - stiffness * Mat3_T::Identity();
        else throw std::runtime_error("d2E_dx2: indices " + std::to_string(i) + ", " + std::to_string(j) + " out of bounds");
    }
    else {
        const Vec3_T xA = position_A->p();
        const Vec3_T xB = position_B->p();
        const Vec3_T d = xA - xB;
        Real_ dist = sqrt(d.dot(d));
        if      ((i == 0 && j == 0) || (i == 1 && j == 1)) return   stiffness * (Mat3_T::Identity() * (1 - rest_length/dist) + (xA - xB) * (xA - xB).transpose() * rest_length/dist/dist/dist);
        else if ((i == 0 && j == 1) || (i == 1 && j == 0)) return - stiffness * (Mat3_T::Identity() * (1 - rest_length/dist) + (xA - xB) * (xA - xB).transpose() * rest_length/dist/dist/dist);
        else throw std::runtime_error("d2E_dx2: indices " + std::to_string(i) + ", " + std::to_string(j) + " out of bounds");
    }
}

template <typename Real_>
Real_  Spring_T<Real_>::d2E_du2(size_t i, size_t j) const {
    if (rest_vars_type != RestVarsType::SpringAnchors && rest_vars_type != RestVarsType::StiffnessAndSpringAnchors)
        throw std::runtime_error("d2E_du2() called for a spring with rest_vars_type != RestVarsType::SpringAnchors or RestVarsType::StiffnessAndSpringAnchors");

    if      (i == 0 && j == 0) return d2E_dxdu(i, j).dot(position_A->dp_du());
    else if (i == 1 && j == 1) return d2E_dxdu(i, j).dot(position_B->dp_du());
    else if (i == 0 && j == 1) return d2E_dxdu(i, j).dot(position_A->dp_du());
    else if (i == 1 && j == 0) return d2E_dxdu(i, j).dot(position_B->dp_du());
    else throw std::runtime_error("d2E_du2: indices " + std::to_string(i) + ", " + std::to_string(j) + " out of bounds");
}

template <typename Real_>
typename Spring_T<Real_>::Vec3_T Spring_T<Real_>::d2E_dxdk(size_t i) const {
    Vec3_T d = position_A->p() - position_B->p();
    Real_ dist = sqrt(d.dot(d));
    if      (i == 0) return   (1 - rest_length/dist) * d;
    else if (i == 1) return - (1 - rest_length/dist) * d;
    else throw std::runtime_error("d2E_dxdk: index " + std::to_string(i) + " out of bounds");
}

template <typename Real_>
typename Spring_T<Real_>::Vec3_T Spring_T<Real_>::d2E_dxdu(size_t xi, size_t ui) const {   
    if (rest_length == 0.0) {
        if       (xi == 0 && ui == 0)  return   stiffness * position_A->dp_du();
        else if  (xi == 1 && ui == 1)  return   stiffness * position_B->dp_du();
        else if ((xi == 0 && ui == 1)) return - stiffness * position_B->dp_du();
        else if ((xi == 1 && ui == 0)) return - stiffness * position_A->dp_du();
        else throw std::runtime_error("d2E_dxdu: indices " + std::to_string(xi) + ", " + std::to_string(ui) + " out of bounds");
    }
    else {
        const Vec3_T xA = position_A->p();
        const Vec3_T xB = position_B->p();
        const Vec3_T d = xA - xB;
        Real_ dist = sqrt(d.dot(d));
        if       (xi == 0 && ui == 0)  return   stiffness * (position_A->dp_du() * (1 - rest_length / dist) + (xA - xB) * (xA - xB).dot(position_A->dp_du()) * rest_length/dist/dist/dist);
        else if  (xi == 1 && ui == 1)  return   stiffness * (position_B->dp_du() * (1 - rest_length / dist) + (xA - xB) * (xA - xB).dot(position_B->dp_du()) * rest_length/dist/dist/dist);
        else if ((xi == 0 && ui == 1)) return - stiffness * (position_B->dp_du() * (1 - rest_length / dist) + (xA - xB) * (xA - xB).dot(position_B->dp_du()) * rest_length/dist/dist/dist);
        else if ((xi == 1 && ui == 0)) return - stiffness * (position_A->dp_du() * (1 - rest_length / dist) + (xA - xB) * (xA - xB).dot(position_A->dp_du()) * rest_length/dist/dist/dist);
    }
}

template <typename Real_>
Real_ Spring_T<Real_>::d2E_dkdu(size_t i) const {
    if (rest_vars_type != RestVarsType::StiffnessAndSpringAnchors)
        throw std::runtime_error("d2E_dkdu() called for a spring with rest_vars_type != RestVarsType::StiffnessAndSpringAnchors");

    Vec3_T d = position_A->p() - position_B->p();
    Real_ dist = sqrt(d.dot(d));
    if      (i == 0) return   (1 - rest_length/dist) * d.dot(position_A->dp_du());
    else if (i == 1) return - (1 - rest_length/dist) * d.dot(position_B->dp_du());
    else throw std::runtime_error("d2E_dkdu: index " + std::to_string(i) + " out of bounds");
}

template<typename Real_>
typename Spring_T<Real_>::Vec6_T Spring_T<Real_>::dE_dx() const {
    Vec6_T res;
    res << dE_dx(0), dE_dx(1);
    return res;
}

template<typename Real_>
typename Spring_T<Real_>::Vec2_T Spring_T<Real_>::dE_du() const {
    Vec2_T res;
    res << dE_du(0), dE_du(1);
    return res;
}

template<typename Real_>
VecX_T<Real_> Spring_T<Real_>::dE_dk() const {
    Vec3_T d = coords_diff();
    Real_ dist = sqrt(d.dot(d));
    Real_ dk = 0.5 * (dist - rest_length) * (dist - rest_length);
    VecX_T res = VecX_T(1);
    res[0] = dk;
    return res;
}

template<typename Real_>
VecX_T<Real_> Spring_T<Real_>::dE_dxk() const {
    Vec3_T d = position_A->p() - position_B->p();
    Real_ dist = sqrt(d.dot(d));
    Real_ dk = 0.5 * (dist - rest_length) * (dist - rest_length);
    VecX_T d2 = VecX_T(6);
    d2 << d, -d;
    VecX_T res(7);
    res.head(6) = stiffness * (1 - rest_length/dist) * d2;
    res[-1] = dk; 
    return res;
}

template<typename Real_>
VecX_T<Real_> Spring_T<Real_>::dE_dxu() const {
    VecX_T res = VecX_T(8);
    res << dE_dx(), dE_du();
    return res;
}

template<typename Real_>
VecX_T<Real_> Spring_T<Real_>::dE_dku() const {
    VecX_T res = VecX_T(3);
    res << dE_dk(), dE_du();
    return res;
}

template<typename Real_>
VecX_T<Real_> Spring_T<Real_>::dE_dxku() const {
    VecX_T res = VecX_T(9);
    res << dE_dx(), dE_dk(), dE_du();
    return res;
}

template <typename Real_>
Eigen::Matrix<Real_, -1, -1> Spring_T<Real_>::d2E_dx2() const {
    Vec3_T d = position_A->p() - position_B->p();

    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast || rest_vars_type == RestVarsType::Stiffness) {

        Eigen::Matrix<Real_, 6, 6> values = Eigen::Matrix<Real_, 6, 6>::Zero();

        if (rest_length != 0.0) {

            Real_   dx = d[0], 
                    dy = d[1],
                    dz = d[2];
            Real_   dx2 = dx * dx, 
                    dy2 = dy * dy, 
                    dz2 = dz * dz;
            Real_   dist3_2 = sqrt((dx2 + dy2 + dz2) * (dx2 + dy2 + dz2) * (dx2 + dy2 + dz2));

            values << 
                // first row
                -(dy2 + dz2),
                dx * dy, 
                dx * dz, 
                dy2 + dz2,
                -dx * dy, 
                -dx * dz,

                // second row
                dx * dy, 
                -(dx2 + dz2), 
                dy * dz, 
                -dx * dy, 
                dx2 + dz2,
                -dy * dz,
            
                // third row
                dx * dz, 
                dy * dz, 
                -(dx2 + dy2),
                -dx * dz, 
                -dy * dz, 
                dx2 + dy2,

                
                // fourth row
                dy2 + dz2,
                -dx * dy, 
                -dx * dz, 
                -(dy2 + dz2),
                dx * dy, 
                dx * dz,
                
                // 5-th row
                -dx * dy, 
                dx2 + dz2,
                -dy * dz, 
                dx * dy, 
                -(dx2 + dz2),
                dy * dz,

                
                // 6-th row
                -dx * dz, 
                -dy * dz, 
                dx2 + dy2,
                dx * dz, 
                dy * dz, 
                -(dx2 + dy2)
            ;
                        
            values *= (stiffness * rest_length / dist3_2);
        }

        for (int i = 0; i < 6; i++) { values(i, i) += stiffness; values(i, (i + 3) % 6) -= stiffness; }

        return values;
    }
    else if (rest_vars_type == RestVarsType::SpringAnchors || rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
        throw std::runtime_error("d2E_dx2() not implemented for rest_vars_type != RestVarsType::StiffnessOnVerticesFast. Call the component-wise version d2E_dx2(size_t, size_t), instead.");  // can be implemented, if needed
    }
    else
        throw std::runtime_error("Unknown rest vars type");
}

template <typename Real_>
typename Spring_T<Real_>::Mat2_T Spring_T<Real_>::d2E_du2() const {
    Mat2_T res;
    res << d2E_du2(0, 0), d2E_du2(0, 1),
           d2E_du2(1, 0), d2E_du2(1, 1);
    return res;
}

template <typename Real_>
VecX_T<Real_> Spring_T<Real_>::d2E_dxdk() const {
    Vec3_T d = position_A->p() - position_B->p();
    Real_ dist = sqrt(d.dot(d));
    VecX_T d2 = VecX_T(6);
    d2 << d, -d;
    return (1 - rest_length/dist) * d2;
}

template <typename Real_>
VecX_T<Real_> Spring_T<Real_>::d2E_dxdu() const {
    throw std::runtime_error("d2E_dxdu() not implemented. Call the component-wise version d2E_dxdu(size_t, size_t), instead.");  // can be implemented, if needed
}

template <typename Real_>
void Spring_T<Real_>::set_rest_vars_type(RestVarsType rvt, bool updateState) { 
    auto old_rvt = rest_vars_type;
    rest_vars_type = rvt;

    if (updateState && old_rvt != rest_vars_type) { 
         // nothing to do here, a Tencer will handle the update
    }
}

template <typename Real_>
VecX_T<Real_> Spring_T<Real_>::get_coords() const {
    VecX_T defo_vars = VecX_T(6);
    defo_vars << position_A->p(), position_B->p();
    return defo_vars;
}

template <typename Real_>
Vec3_T<Real_> Spring_T<Real_>::coords_diff() const{
    return position_A->p() - position_B->p();
}




 
template struct Spring_T<Real>;
template struct Spring_T<ADReal>;