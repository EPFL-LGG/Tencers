#ifndef TencerOPTIMIZATION_HH
#define TencerOPTIMIZATION_HH

#include <DesignOptimization/EquilibriumOptimizationObject.hh>
#include <MeshFEM/newton_optimizer/newton_optimizer.hh>

#include "Tencer.hh"
#include "weighted_target_curve_fitter.hh"

enum class OptEnergyType   : int {Full, CurveFitting};

/*********************************************
 **********   OPTIMIZATION OBJECT   **********
 *********************************************/

struct TencerOptimization : public EquilibriumOptimizationObject<Tencer_T,OptEnergyType> {

    using Tencer = Tencer_T<Real>;
    using ADTencer = Tencer_T<ADReal>;
    using KnotEnergyType = Tencer::KnotEnergyType;
    
    TencerOptimization(Tencer &knot, const NewtonOptimizerOptions &eopts, const std::vector<size_t> &fixedVars, std::vector<ElasticRod> &open_target_rods, std::vector<PeriodicRod> &closed_target_rods, std::vector<VecX> &radii,VecX weight = {}, Real hessian_shift = 1e-6, PredictionOrder po = PredictionOrder::One);
    TencerOptimization(TencerOptimization &sp);

    void update_target_rods(std::vector<ElasticRod> &open_target_rods,std::vector<PeriodicRod> &closed_target_rods,std::vector<VecX> &radii);

    virtual Real J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType = OptEnergyType::Full) override;
    virtual Eigen::VectorXd gradp_J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType = OptEnergyType::Full) override;
    Eigen::VectorXd dJ_dp(OptEnergyType opt_eType = OptEnergyType::Full); // computes gradient without the need to recompute adjoint state
    virtual Eigen::VectorXd apply_hess(const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, Real coeff_J, OptEnergyType opt_eType = OptEnergyType::Full) override;

    size_t numRestVars() {return m_linesearch_object.numRestVars();}
    virtual const Eigen::VectorXd params() const override{ 
        return is_non_zero_weight.cwiseProduct(m_object.getRestVars().array().pow(r).matrix()) + is_zero_weight.cwiseProduct(m_object.getRestVars()); 
    }

    void set_norm_param(Real param) {
        assert((0 < param) & (1 > param));
        r = param;
    }

    VecX apply_change_variable(const VecX &params, const VecX &v);
    VecX apply_change_variable_second_order(const VecX &params, const VecX &delta_params, const VecX &v);
    
    virtual void newPt(const Eigen::VectorXd &params) override;

    std::vector<Real> get_vars_lower_bounds() const { return m_object.get_rest_vars_lower_bounds(); }
    std::vector<Real> get_vars_upper_bounds() const { return m_object.get_rest_vars_upper_bounds(); }

    Real hess_shift;

    virtual bool m_updateEquilibria(const Eigen::Ref<const Eigen::VectorXd> &newParams) override;



protected:
    Real r;
    VecX sparsification_weight;
    VecX is_zero_weight;
    VecX is_non_zero_weight;
    bool optimize_spring_positions;

};




/*************************************************************
 **********   DESIGN OPTIMIZATION OBJECTIVE TERMS   **********
 *************************************************************/

// Wrapper for the weighted target curve fitting objective optimization term.
struct WeightedTargetFittingDOOT : public DesignOptimizationObjectiveTerm<Tencer_T> {
    using DOT      = DesignOptimizationTerm<Tencer_T>;
    using DOOT     = DesignOptimizationObjectiveTerm<Tencer_T>;
    using Tencer   = typename DOT::Object;
    using ADTencer = typename DOT::ADObject;
    using VecX   = VecX_T<Real>;

    WeightedTargetFittingDOOT(Tencer &obj, std::vector<ElasticRod> &open_targets, std::vector<PeriodicRod> &closed_targets, std::vector<VecX> &rad): DOOT(obj) {
        for (size_t i = 0; i < open_targets.size(); ++i){
            wtcf.push_back(WeightedTargetCurveFitter<Real>(open_targets[i],rad[i]));
        }
        for (size_t i = 0; i < closed_targets.size(); ++i){
            size_t idx = open_targets.size() + i;
            wtcf.push_back(WeightedTargetCurveFitter<Real>(closed_targets[i],rad[idx]));
        }
    } 

    using DOOT::weight;
protected:
    using DOT::m_obj;
    std::vector<WeightedTargetCurveFitter<Real>> wtcf;

    //////////////////////////////////////////////////////
    // Implementation of DesignOptimizationTerm interface
    //////////////////////////////////////////////////////

    // nothing to update
    virtual void m_update() override {}
    
    // J(x,p)
    virtual Real m_value() const override { 
        Real e = 0;
        for (size_t i = 0; i < m_obj.num_open_rods(); ++i){
            e += wtcf[i].energy(m_obj.open_rods[i]);
        }
        for (size_t i = 0; i < m_obj.num_closed_rods(); ++i){
            int idx = m_obj.num_open_rods() + i;
            e += wtcf[idx].energy(m_obj.closed_rods[i]);
        }
        return e;
    }

    // [dJ/dx, dJ/dp]
    virtual VecX m_grad() const override {
        VecX g = VecX::Zero(m_obj.numVars(VariableMask::All));
        for (size_t i = 0; i < m_obj.num_open_rods(); ++i){
            g.segment(m_obj.get_defo_var_index_for_rod(i), m_obj.get_num_defo_var_for_rod(i)) = wtcf[i].grad(m_obj.open_rods[i]);
        }
        for (size_t i = 0; i < m_obj.num_closed_rods(); ++i){
            int idx = m_obj.num_open_rods() + i;
            g.segment(m_obj.get_defo_var_index_for_rod(idx), m_obj.get_num_defo_var_for_rod(idx)) = wtcf[idx].grad(m_obj.closed_rods[i]);
        }
        return g;
    }

    // (d^2 J / d{x, p} d{x, p}) [ delta_x, delta_p ]
    virtual VecX m_delta_grad(Eigen::Ref<const VecX> delta_xp, const ADTencer &/* ado */) const override {
        VecX res = VecX::Zero(m_obj.numVars(VariableMask::All));
        for (size_t i = 0; i < m_obj.num_open_rods(); ++i){
            wtcf[i].apply_hessian(m_obj.open_rods[i],delta_xp,res,1.0,m_obj.get_defo_var_index_for_rod(i));
        }
        for (size_t i = 0; i < m_obj.num_closed_rods(); ++i){
            int idx = m_obj.num_open_rods() + i;
            wtcf[idx].apply_hessian(m_obj.closed_rods[i],delta_xp,res,1.0,m_obj.get_defo_var_index_for_rod(idx));
        }
        return res;
    }
};


#endif /* end of include guard: TencerOPTIMIZATION_HH */