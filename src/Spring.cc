#include "Spring.hh"

template <typename Real_>
Spring_T<Real_>::Spring_T(Vec3_T &pA, Vec3_T &pB, Real_ s, Real_ l,CompressionType c, Real tol) 
    : position_A(std::make_shared<FixedNode_T<Real_>>(pA)), 
      position_B(std::make_shared<FixedNode_T<Real_>>(pB)), 
      stiffness(s), rest_length(l),
      compression_type(c), compression_tolerance(tol) 
      {  }

template <typename Real_>
Spring_T<Real_>::Spring_T(const std::shared_ptr<Node_T<Real_>> &pA, const std::shared_ptr<Node_T<Real_>> &pB, Real_ s, Real_ l,CompressionType c, Real tol) 
    : position_A(pA), position_B(pB), stiffness(s), rest_length(l), compression_type(c), compression_tolerance(tol) 
    {  }

template <typename Real_>
Real_ Spring_T<Real_>::energy() const{
    Vec3_T d = position_A->p() - position_B->p();
    Real_ dist = sqrt(d.dot(d));
    Real_ x = dist - rest_length;
    if (compression_type == CompressionType::Compression){
        return 0.5 * stiffness * x*x;
    }
    else {
        return 0.5 * stiffness * Q(x);
    }
}


template<typename Real_>
typename Spring_T<Real_>::Vec3_T Spring_T<Real_>::dE_dx(size_t i) const {
    Vec3_T d = position_A->p() - position_B->p();
    Real_ dist = sqrt(d.dot(d));
    if (compression_type == CompressionType::Compression){
        if      (i == 0) return   stiffness * (1 - rest_length/dist) * d;
        else if (i == 1) return - stiffness * (1 - rest_length/dist) * d;
        else throw std::runtime_error("dE_dx: index " + std::to_string(i) + " out of bounds");
    }
    else {
        Real_ x = dist - rest_length;
        if      (i == 0) return   0.5 * stiffness * dQ_dx(x) * d / dist;
        else if (i == 1) return - 0.5 * stiffness * dQ_dx(x) * d / dist;
        else throw std::runtime_error("dE_dx: index " + std::to_string(i) + " out of bounds");
    }
}

template<typename Real_>
Real_ Spring_T<Real_>::dE_du(size_t i) const {
    if (compression_type == CompressionType::Compression){
        if      (i == 0) return dE_dx(0).dot(position_A->dp_du());
        else if (i == 1) return dE_dx(1).dot(position_B->dp_du());
        else throw std::runtime_error("dE_du: index " + std::to_string(i) + " out of bounds");
    }
    else throw std::runtime_error("not implemented for NoCompression type.");
}

template <typename Real_>
typename Spring_T<Real_>::Mat3_T Spring_T<Real_>::d2E_dx2(size_t i, size_t j) const {
    if (compression_type == CompressionType::Compression){
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
    else throw std::runtime_error("not implemented for NoCompression type.");
}

template <typename Real_>
Real_  Spring_T<Real_>::d2E_du2(size_t i, size_t j) const {
    if (compression_type == CompressionType::Compression){
        if (rest_length != 0.0)
            throw std::runtime_error("d2E_du2() not implemented for rest_length != 0.0");  // TODO MICHELE

        if       (i == 0 && j == 0)                        return   stiffness * position_A->dp_du().dot(position_A->dp_du());
        else if  (i == 1 && j == 1)                        return   stiffness * position_B->dp_du().dot(position_B->dp_du());
        else if ((i == 0 && j == 1) || (i == 1 && j == 0)) return - stiffness * position_A->dp_du().dot(position_B->dp_du());
        else throw std::runtime_error("d2E_du2: indices " + std::to_string(i) + ", " + std::to_string(j) + " out of bounds");
    }
    else throw std::runtime_error("not implemented for NoCompression type.");
}

template <typename Real_>
typename Spring_T<Real_>::Vec3_T Spring_T<Real_>::d2E_dxdk(size_t i) const {
    if (compression_type == CompressionType::Compression){
        if (rest_vars_type != RestVarsType::StiffnessOnVerticesFast) {
            Vec3_T d = position_A->p() - position_B->p();
            Real_ dist = sqrt(d.dot(d));
            if      (i == 0) return   (1 - rest_length/dist) * d;
            else if (i == 1) return - (1 - rest_length/dist) * d;
            else throw std::runtime_error("d2E_dxdk: index " + std::to_string(i) + " out of bounds");
        }
        else throw std::runtime_error("d2E_dxdk not implemented for RestVarsType::StiffnessOnVerticesFast"); //TODO : is this correct? I added it because of a warning
    }
    else throw std::runtime_error("not implemented for NoCompression type.");
}


template <typename Real_>
typename Spring_T<Real_>::Vec3_T Spring_T<Real_>::d2E_dxdu(size_t xi, size_t ui) const {
    if (compression_type == CompressionType::Compression){
        if (rest_length != 0.0)
            throw std::runtime_error("d2E_dxdu() not implemented for rest_length != 0.0");  // TODO MICHELE: rest_length != 0.0 would need a full matrix to be returned, the function signature would need to change
        
        if       (xi == 0 && ui == 0)  return   stiffness * position_A->dp_du();
        else if  (xi == 1 && ui == 1)  return   stiffness * position_B->dp_du();
        else if ((xi == 0 && ui == 1)) return - stiffness * position_B->dp_du();
        else if ((xi == 1 && ui == 0)) return - stiffness * position_A->dp_du();
        else throw std::runtime_error("d2E_dxdu: indices " + std::to_string(xi) + ", " + std::to_string(ui) + " out of bounds");
    }
    else throw std::runtime_error("not implemented for NoCompression type.");
}

template <typename Real_>
Real_ Spring_T<Real_>::d2E_dkdu(size_t i) const {
    if (compression_type == CompressionType::Compression){
        if (rest_vars_type != RestVarsType::StiffnessAndSpringAnchors)
            throw std::runtime_error("d2E_dkdu() called for a spring with rest_vars_type != RestVarsType::StiffnessAndSpringAnchors");

        Vec3_T d = position_A->p() - position_B->p();
        Real_ dist = sqrt(d.dot(d));
        if      (i == 0) return   (1 - rest_length/dist) * d.dot(position_A->dp_du());
        else if (i == 1) return - (1 - rest_length/dist) * d.dot(position_B->dp_du());
        else throw std::runtime_error("d2E_dkdu: index " + std::to_string(i) + " out of bounds");
    }
    else throw std::runtime_error("not implemented for NoCompression type.");
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
    if (compression_type == CompressionType::Compression){
        Vec3_T d = coords_diff();
        Real_ dist = sqrt(d.dot(d));
        Real_ dk = 0.5 * (dist - rest_length) * (dist - rest_length);
        VecX_T res = VecX_T(1);
        res[0] = dk;
        return res;
    }
    else throw std::runtime_error("not implemented for NoCompression type.");
}

template<typename Real_>
VecX_T<Real_> Spring_T<Real_>::dE_dxk() const {
    if (compression_type == CompressionType::Compression){
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
    else throw std::runtime_error("not implemented for NoCompression type.");
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
    Real_ dist = sqrt(d.dot(d));

    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast || rest_vars_type == RestVarsType::Stiffness) {

        Eigen::Matrix<Real_, 6, 6> values = Eigen::Matrix<Real_, 6, 6>::Zero();
        Real_ L3 = dist*dist*dist;

        if (compression_type == CompressionType::Compression){
            if (rest_length != 0.0) {
                values.block(0,0,3,3) =  d * d.transpose();
                values.block(3,0,3,3) = -d * d.transpose();
                values.block(0,3,3,3) = -d * d.transpose();
                values.block(3,3,3,3) =  d * d.transpose();
                values *= (stiffness * rest_length / L3);
                for (int i = 0; i < 6; i++) { values(i, i) -= stiffness*rest_length/dist; values(i, (i + 3) % 6) += stiffness*rest_length/dist; }
            }
            for (int i = 0; i < 6; i++) { values(i, i) += stiffness; values(i, (i + 3) % 6) -= stiffness; }
        }
        else{
            Real_ x = dist - rest_length;
            Real_ L2 = dist*dist;
            Real_ dQ_dx_L_L0 = dQ_dx(x);
            values.block(0,0,3,3) =  d * d.transpose();
            values.block(3,0,3,3) = -d * d.transpose();
            values.block(0,3,3,3) = -d * d.transpose();
            values.block(3,3,3,3) =  d * d.transpose();
            values *= 0.5 * stiffness;
            values *= d2Q_dx2(x)/L2 - dQ_dx_L_L0/L3;

            values.block(0,0,3,3) += 0.5 * stiffness * dQ_dx_L_L0 / dist * Eigen::Matrix<Real_, 3, 3>::Identity();
            values.block(3,0,3,3) -= 0.5 * stiffness * dQ_dx_L_L0 / dist * Eigen::Matrix<Real_, 3, 3>::Identity();
            values.block(0,3,3,3) -= 0.5 * stiffness * dQ_dx_L_L0 / dist * Eigen::Matrix<Real_, 3, 3>::Identity();
            values.block(3,3,3,3) += 0.5 * stiffness * dQ_dx_L_L0 / dist * Eigen::Matrix<Real_, 3, 3>::Identity();
        }
        return values;
    }
    else if (rest_vars_type == RestVarsType::SpringAnchors || rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
        throw std::runtime_error("d2E_dx2() not implemented for rest_vars_type != RestVarsType::StiffnessOnVerticesFast. Call d2E_dx2(size_t, size_t), instead.");  // TODO MICHELE
    }
    else
        throw std::runtime_error("Unknown rest vars type");
}

template <typename Real_>
typename Spring_T<Real_>::Mat2_T Spring_T<Real_>::d2E_du2() const {
    if (compression_type == CompressionType::Compression){
        if (rest_length != 0.0)
            throw std::runtime_error("d2E_du2() not implemented for rest_length != 0.0");  // TODO MICHELE
        Mat2_T res;
        res <<   position_A->dp_du().dot(position_A->dp_du()), - position_A->dp_du().dot(position_B->dp_du()),
            - position_B->dp_du().dot(position_A->dp_du()),   position_B->dp_du().dot(position_B->dp_du());
        res *= stiffness;
        return res;
    }
    else throw std::runtime_error("not implemented for NoCompression type.");
}

template <typename Real_>
VecX_T<Real_> Spring_T<Real_>::d2E_dxdk() const {
    if (compression_type == CompressionType::Compression){
        Vec3_T d = position_A->p() - position_B->p();
        Real_ dist = sqrt(d.dot(d));
        VecX_T d2 = VecX_T(6);
        d2 << d, -d;
        return (1 - rest_length/dist) * d2;
    }
    else throw std::runtime_error("not implemented for NoCompression type.");
}

template <typename Real_>
VecX_T<Real_> Spring_T<Real_>::d2E_dxdu() const {
    throw std::runtime_error("d2E_dxdu() not implemented");  
}

template <typename Real_>
VecX_T<Real_> Spring_T<Real_>::d2E_dxdl0() const {
    Vec3_T d = position_A->p() - position_B->p();
    Real_ L = sqrt(d.dot(d));

    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast || rest_vars_type == RestVarsType::Stiffness) {

        VecX_T values = VecX_T::Zero(6);

        if (compression_type == CompressionType::Compression){
            values.head(3) =  - stiffness * d / L;
            values.tail(3) = stiffness * d / L;
        }
        else{
            Real_ x = L - rest_length;
            Real_ dQ_dx_dL0 = -d2Q_dx2(x);
            values.head(3) =  0.5 * stiffness * dQ_dx_dL0 * d / L;
            values.tail(3) = -values.head(3);    
        }
        return values;
    }
    else if (rest_vars_type == RestVarsType::SpringAnchors || rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
        throw std::runtime_error("d2E_dxdl() not implemented for rest_vars_type != RestVarsType::StiffnessOnVerticesFast."); 
    }
    else
        throw std::runtime_error("Unknown rest vars type");
}

template <typename Real_>
Real_ Spring_T<Real_>::dE_dl0() const {
    Vec3_T d = position_A->p() - position_B->p();
    Real_ L = sqrt(d.dot(d));
    Real_ x = L - rest_length;

    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast || rest_vars_type == RestVarsType::Stiffness) {
        Real_ result = 0;
        if (compression_type == CompressionType::Compression){
            result =  - stiffness * x;
        }
        else{
            Real_ dQ_dL0 = -dQ_dx(x);
            result = 0.5 * stiffness * dQ_dL0;
        }
        return result;
    }
    else if (rest_vars_type == RestVarsType::SpringAnchors || rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
        throw std::runtime_error("d2E_dxdl() not implemented for rest_vars_type != RestVarsType::StiffnessOnVerticesFast."); 
    }
    else
        throw std::runtime_error("Unknown rest vars type");
}

template <typename Real_>
Real_ Spring_T<Real_>::d2E_dl02() const {
    if (rest_vars_type == RestVarsType::StiffnessOnVerticesFast || rest_vars_type == RestVarsType::Stiffness) {
        Real_ result = 0;
        if (compression_type == CompressionType::Compression){
            result =  stiffness;
        }
        else{
            Vec3_T d = position_A->p() - position_B->p();
            Real_ L = sqrt(d.dot(d));
            Real_ x = L - rest_length;
            Real_ d2Q_dL02 = d2Q_dx2(x);
            result = 0.5 * stiffness * d2Q_dL02;
        }
        return result;
    }
    else if (rest_vars_type == RestVarsType::SpringAnchors || rest_vars_type == RestVarsType::StiffnessAndSpringAnchors) {
        throw std::runtime_error("d2E_dxdl() not implemented for rest_vars_type != RestVarsType::StiffnessOnVerticesFast."); 
    }
    else
        throw std::runtime_error("Unknown rest vars type");
}


template <typename Real_>
Real_ Spring_T<Real_>::Q(Real_ x) const{
    if (x < -compression_tolerance) return 0;
    if (x < compression_tolerance){
        return x*x*x/(6*compression_tolerance) + x*x/2 + compression_tolerance*x/2 + compression_tolerance*compression_tolerance/6;
    }
    return x*x + compression_tolerance*compression_tolerance/3;
}

template <typename Real_>
Real_ Spring_T<Real_>::dQ_dx(Real_ x) const{
    if (x < -compression_tolerance) return 0;
    if (x < compression_tolerance){
        return x*x/(2*compression_tolerance) + x + compression_tolerance/2;
    }
    return 2*x;
}

template <typename Real_>
Real_ Spring_T<Real_>::d2Q_dx2(Real_ x) const{
    if (x < -compression_tolerance) return 0;
    if (x < compression_tolerance){
        return x/compression_tolerance + 1;
    }
    return 2;
}

template <typename Real_>
void Spring_T<Real_>::set_rest_vars_type(RestVarsType rvt, bool updateState) { 
    auto old_rvt = rest_vars_type;
    rest_vars_type = rvt;

    if (updateState && old_rvt != rest_vars_type) { 
         // TODO MICHELE: do we want to 1) implement this here and stop delegating the update to TensegrityKnot, or 2) restructure class members to avoid this side effect?
    }
}

template <typename Real_>
VecX_T<Real_> Spring_T<Real_>::get_coords() const {
    VecX_T defo_vars = VecX_T(6);
    defo_vars << position_A->p(), position_B->p();
    return defo_vars;
}

template <typename Real_>
void Spring_T<Real_>::set_coords(VecX_T coords) {
    position_A->set(coords.head(3));
    position_B->set(coords.tail(3));
}

template <typename Real_>
Vec3_T<Real_> Spring_T<Real_>::coords_diff() const {
    return position_A->p() - position_B->p();
}

template <typename Real_>
Real_ Spring_T<Real_>::force_norm() const {
    Vec3_T d = position_A->p() - position_B->p();
    Real_ dist = sqrt(d.dot(d));
    Real_ x = dist - rest_length;
    if (compression_type == CompressionType::Compression){
        return stiffness * abs(x);
    }
    else {
        return 0.5 * stiffness * dQ_dx(x);
    }
}

template <typename Real_>
Real_ Spring_T<Real_>::d_force_norm_dL0() const {
    Vec3_T d = position_A->p() - position_B->p();
    Real_ dist = sqrt(d.dot(d));
    Real_ x = dist - rest_length;
    if (compression_type == CompressionType::Compression){
        Real_ sgn = (x > 0) ? 1.0 : -1.0;
        return - sgn * stiffness;
    }
    else {
        return - 0.5 * stiffness * d2Q_dx2(x);
    }
}

template <typename Real_>
VecX_T<Real_> Spring_T<Real_>::d_force_norm_dx() const{
    Vec3_T d = position_A->p() - position_B->p();
    Real_ dist = sqrt(d.dot(d));
    VecX_T result = VecX_T::Zero(6);
    Real_ x = dist - rest_length;
    Vec3_T g;
    if (compression_type == CompressionType::Compression){
        Real_ sgn = (x > 0) ? 1.0 : -1.0;
        g = sgn * stiffness * d / dist;
        result.head(3) = g;
        result.tail(3) = - g;
    }
    else {
        g = 0.5 * stiffness * d / dist * d2Q_dx2(x);
        result.head(3) = g;
        result.tail(3) = - g;
    }
    return result;
}


 
template struct Spring_T<Real>;
template struct Spring_T<ADReal>;