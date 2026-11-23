#![forbid(unsafe_code)]

//! Mechanistic IVIVE → Minimal-PBPK & Human Therapeutic Dose Simulator (`pbpk`).
//!
//! Combines local calibrated TDC QSAR predictions (`ppbr_az`, `clearance_microsome_az`,
//! `vdss_lombardo`, `half_life_obach`, `caco2_wang`, `hia_hou`, `bioavailability_ma`,
//! `solubility_aqsoldb`, `bbb_martins`, `pgp_broccatelli`, `cyp3a4_veith`, `herg_karim`)
//! with Austin (2002) microsomal binding, the well-stirred liver model, Amidon/Sun
//! permeability-solubility oral absorption (BCS Class I–IV + Dose Number `Do`), a
//! mass-conserving 4th-order Runge-Kutta (`RK4`) multi-compartment ODE engine (`Oral | IV | SC`,
//! with optional saturable TMDD clearance hook), and 200-draw deterministic Monte Carlo propagation
//! of the 90% Split Conformal QSAR intervals.

use std::fmt::Write as _;

use admet_calc::{
    BcsClass, HergRisk, IonizationClass, MoleculeEvaluation, PbpkHumanDoseReport, PkRoute,
    PkSimulationConfig, PkTimePoint, QsarAdmetReport, RegimenDoseEstimate,
};

const Q_H_ML_MIN_KG: f64 = 20.7; // Human hepatic blood flow (1.45 L/min / 70 kg)
const MPPGL_MG_PER_G: f64 = 40.0; // Microsomal protein per gram human liver
const LIVER_G_PER_KG: f64 = 25.7; // 1800 g liver / 70 kg adult
const GFR_ML_MIN_KG: f64 = 1.8; // 126 mL/min / 70 kg adult
const C_MIC_MG_ML: f64 = 0.5; // Typical microsomal incubation protein conc.
const RT_298_KCAL: f64 = 0.592_186; // R * 298.15 K in kcal/mol

/// Converts a binding free energy `ΔG` (kcal/mol) at 298.15 K to `Ki` in nanomolar (`nM`).
#[must_use]
pub fn dg_kcal_to_ki_nm(dg_kcal: f64) -> f64 {
    let kd_m = (dg_kcal / RT_298_KCAL).exp();
    (kd_m * 1.0e9).clamp(1.0e-3, 1.0e7)
}

#[derive(Debug, Clone, Copy)]
struct PkSampleParams {
    fup: f64,
    fu_inc: f64,
    rb: f64,
    cl_int_u: f64,
    cl_h_ml_min_kg: f64,
    cl_r_ml_min_kg: f64,
    cl_tot_ml_min_kg: f64,
    eh: f64,
    fh: f64,
    vdss_l_kg: f64,
    peff_1e4: f64,
    sol_mg_ml: f64,
    fg: f64,
    ka_per_h: f64,
    kpuu_brain: f64,
    frac_central: f64,
}

fn qsar_ep<'a>(
    qsar: Option<&'a QsarAdmetReport>,
    id: &str,
) -> Option<&'a admet_calc::QsarEndpointPrediction> {
    qsar.and_then(|q| q.endpoints.iter().find(|e| e.endpoint_id == id))
}

/// Like [`qsar_ep`], but records the endpoint id when no trained prediction exists, so the
/// report can say which of its inputs are correlations rather than model output.
fn qsar_ep_tracked<'a>(
    qsar: Option<&'a QsarAdmetReport>,
    id: &str,
    estimated: &mut Vec<String>,
) -> Option<&'a admet_calc::QsarEndpointPrediction> {
    let found = qsar_ep(qsar, id);
    if found.is_none() {
        estimated.push(id.to_string());
    }
    found
}

struct BaselineQsarInputs {
    /// Endpoint ids that fell back to a logD correlation because the QSAR pack did not
    /// provide them. Empty means every input below came from a trained model.
    estimated: Vec<String>,
    ppbr: (f64, f64, f64),
    cl_mic: (f64, f64, f64),
    vdss: (f64, f64, f64),
    caco2: (f64, f64, f64),
    sol_mg_ml: f64,
    t_half_obach: Option<f64>,
    cyp3a4: f64,
    pgp: f64,
    bbb: f64,
    hia: f64,
    herg: f64,
}

fn extract_baseline_and_ci90(eval: &MoleculeEvaluation) -> BaselineQsarInputs {
    let qsar = eval.qsar.as_ref();
    let mut estimated: Vec<String> = Vec::new();
    let logd = eval.ionization.logd_ph74;
    let mw = eval.physicochemical.molecular_weight.max(50.0);

    // 1. PPBR (%)
    let ppbr = if let Some(ep) = qsar_ep_tracked(qsar, "ppbr_az", &mut estimated) {
        let med = ep.value.clamp(1.0, 99.5);
        let lo = ep.ci90_low.clamp(1.0, med);
        let hi = ep.ci90_high.clamp(med, 99.5);
        (med, lo, hi)
    } else {
        let fup_est = (1.0 / (1.0 + 10.0_f64.powf(0.45 * logd - 0.35))).clamp(0.01, 0.98);
        let med = (1.0 - fup_est) * 100.0;
        (med, (med - 12.0).max(1.0), (med + 10.0).min(99.5))
    };

    // 2. Microsomal CLint (uL/min/mg)
    let cl_mic = if let Some(ep) = qsar_ep_tracked(qsar, "clearance_microsome_az", &mut estimated) {
        let med = ep.linear_value.unwrap_or(20.0).clamp(1.0, 1500.0);
        let ci = ep.linear_ci90.unwrap_or([med * 0.4, med * 2.5]);
        (med, ci[0].clamp(0.5, med), ci[1].clamp(med, 2500.0))
    } else {
        let med = (12.0 * 10.0_f64.powf((0.28 * logd).clamp(-0.8, 1.4))).clamp(2.0, 400.0);
        (med, med * 0.4, med * 2.5)
    };

    // 3. VDss (L/kg)
    let vdss = if let Some(ep) = qsar_ep_tracked(qsar, "vdss_lombardo", &mut estimated) {
        let med = ep.linear_value.unwrap_or(1.2).clamp(0.08, 40.0);
        let ci = ep.linear_ci90.unwrap_or([med * 0.45, med * 2.2]);
        (med, ci[0].clamp(0.06, med), ci[1].clamp(med, 60.0))
    } else {
        let med = match eval.ionization.ionization_class {
            IonizationClass::Acid => (0.18 + 0.08 * logd.max(0.0)).clamp(0.10, 1.2),
            IonizationClass::Base => {
                (1.5 + 0.7 * 10.0_f64.powf((0.25 * logd).clamp(-0.5, 1.0))).clamp(0.5, 20.0)
            }
            IonizationClass::Neutral | IonizationClass::Zwitterion => {
                (0.55 + 0.35 * 10.0_f64.powf((0.22 * logd).clamp(-0.6, 0.9))).clamp(0.2, 8.0)
            }
        };
        (med, (med * 0.45).max(0.08), (med * 2.2).min(45.0))
    };

    // 4. Caco-2 (log10 cm/s)
    let caco2 = if let Some(ep) = qsar_ep_tracked(qsar, "caco2_wang", &mut estimated) {
        let med = ep.value.clamp(-7.2, -3.8);
        let lo = ep.ci90_low.clamp(-7.5, med);
        let hi = ep.ci90_high.clamp(med, -3.5);
        (med, lo, hi)
    } else {
        let med = (-4.5
            - 0.012 * eval.physicochemical.tpsa
            - 0.12 * eval.physicochemical.h_bond_donors as f64)
            .clamp(-6.8, -4.2);
        (med, (med - 0.45).max(-7.2), (med + 0.45).min(-3.8))
    };

    // 5. Physiological intestinal (FaSSIF pH 6.5) solubility (mg/mL):
    // Intrinsic neutral aqueous solubility (AqSolDB / ESOL) adjusted via Henderson-Hasselbalch
    // ionization at duodenal pH 6.5 (capped at 50x for common-ion/salt limit) and Mithani/Glomme
    // FaSSIF bile-salt micellar solubilization (3 mM taurocholate, capped at 20x; total cap 100x).
    let sol_water_mg_ml =
        if let Some(ep) = qsar_ep_tracked(qsar, "solubility_aqsoldb", &mut estimated) {
            let mol_l = 10.0_f64.powf(ep.value.clamp(-9.0, 1.0));
            (mol_l * mw).clamp(1.0e-5, 1000.0)
        } else {
            eval.water_solubility
                .esol
                .solubility_mg_ml
                .clamp(1.0e-5, 1000.0)
        };
    let acid_ion = eval
        .ionization
        .strongest_acidic_pka
        .map_or(0.0, |pka| 10.0_f64.powf((6.5 - pka).clamp(-6.0, 1.7)));
    let base_ion = eval
        .ionization
        .strongest_basic_pka
        .map_or(0.0, |pka| 10.0_f64.powf((pka - 6.5).clamp(-6.0, 1.7)));
    let ion_boost = (1.0 + acid_ion + base_ion).clamp(1.0, 50.0);
    let logp = eval.lipophilicity.consensus_log_p;
    let fassif_micelle = (1.0 + 10.0_f64.powf(0.55 * logp - 1.35)).clamp(1.0, 20.0);
    let total_gi_boost = (ion_boost * fassif_micelle).min(100.0);
    let sol_mg_ml = (sol_water_mg_ml * total_gi_boost).clamp(1.0e-5, 1000.0);

    let half_life_obach = qsar_ep_tracked(qsar, "half_life_obach", &mut estimated)
        .and_then(|e| e.linear_value)
        .map(|v| v.clamp(0.25, 240.0));

    let cyp3a4 = qsar_ep_tracked(qsar, "cyp3a4_veith", &mut estimated)
        .map(|e| e.value)
        .unwrap_or(if eval.pharmacokinetics.cyp3a4_inhibitor {
            0.7
        } else {
            0.2
        });
    let pgp = qsar_ep_tracked(qsar, "pgp_broccatelli", &mut estimated)
        .map(|e| e.value)
        .unwrap_or(if eval.pharmacokinetics.pgp_substrate {
            0.7
        } else {
            0.2
        });
    let bbb = qsar_ep_tracked(qsar, "bbb_martins", &mut estimated)
        .map(|e| e.value)
        .unwrap_or(if eval.pharmacokinetics.bbb_permeant {
            0.8
        } else {
            0.2
        });
    let hia = qsar_ep_tracked(qsar, "hia_hou", &mut estimated)
        .map(|e| e.value)
        .unwrap_or(match eval.pharmacokinetics.gi_absorption {
            admet_calc::AbsorptionLevel::High => 0.85,
            admet_calc::AbsorptionLevel::Low => 0.35,
        });
    let herg = qsar_ep_tracked(qsar, "herg_karim", &mut estimated)
        .map(|e| e.value)
        .unwrap_or(match eval.safety.herg_risk {
            HergRisk::Low => 0.15,
            HergRisk::Moderate => 0.50,
            HergRisk::High => 0.85,
        });

    BaselineQsarInputs {
        estimated,
        ppbr,
        cl_mic,
        vdss,
        caco2,
        sol_mg_ml,
        t_half_obach: half_life_obach,
        cyp3a4,
        pgp,
        bbb,
        hia,
        herg,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_pk_params(
    eval: &MoleculeEvaluation,
    ppbr_pct: f64,
    cl_mic: f64,
    vdss_l_kg: f64,
    caco2_log: f64,
    sol_mg_ml: f64,
    cyp3a4_p: f64,
    pgp_p: f64,
    bbb_p: f64,
) -> PkSampleParams {
    let logd = eval.ionization.logd_ph74;
    let fup = ((100.0 - ppbr_pct) / 100.0).clamp(0.005, 0.99);
    // Austin (2002) microsomal nonspecific binding correction
    let fu_inc = (1.0 / (1.0 + C_MIC_MG_ML * 10.0_f64.powf(0.53 * logd - 1.42))).clamp(0.03, 0.99);
    let (rb, frac_central) = match eval.ionization.ionization_class {
        IonizationClass::Acid => (0.55, 0.55),
        IonizationClass::Base => (1.00, 0.25),
        IonizationClass::Neutral | IonizationClass::Zwitterion => (0.85, 0.38),
    };
    let fub = (fup / rb).clamp(0.005, 1.0);

    // Unbound intrinsic clearance scaled to whole human liver (mL/min/kg)
    let cl_int_u = (cl_mic / fu_inc).clamp(0.5, 4000.0);
    let cl_int_vivo = cl_int_u * MPPGL_MG_PER_G * LIVER_G_PER_KG / 1000.0;
    let cl_h_blood = (Q_H_ML_MIN_KG * fub * cl_int_vivo) / (Q_H_ML_MIN_KG + fub * cl_int_vivo);
    let eh = (cl_h_blood / Q_H_ML_MIN_KG).clamp(0.0, 0.97);
    let fh = (1.0 - eh).clamp(0.03, 1.0);
    let cl_h_plasma = cl_h_blood * rb;

    // Passive glomerular filtration with lipophilicity-dependent tubular reabsorption
    let f_reabs = 1.0 / (1.0 + (-0.8 * (logd - 0.5)).exp());
    let cl_r_plasma = fup * GFR_ML_MIN_KG * (1.0 - f_reabs);
    let cl_tot = (cl_h_plasma + cl_r_plasma).max(0.05);

    // Sun (2002) Caco-2 -> Human jejunal Peff (10^-4 cm/s)
    let log_peff = 0.6836 * (caco2_log + 6.0) - 0.5579;
    let peff_1e4 = 10.0_f64.powf(log_peff).clamp(0.05, 12.0);
    let fg = (1.0 - 0.25 * cyp3a4_p - 0.18 * pgp_p).clamp(0.35, 0.99);
    let ka_per_h = (2.0 * peff_1e4 * 1.0e-4 * 3600.0 / 1.75).clamp(0.25, 3.2);

    // Unbound brain-to-plasma partition Kp,uu,brain
    let base_kpuu = (0.04 + 0.85 * bbb_p) * (1.0 - 0.65 * pgp_p);
    let kpuu_brain = base_kpuu.clamp(0.01, 1.1);

    PkSampleParams {
        fup,
        fu_inc,
        rb,
        cl_int_u,
        cl_h_ml_min_kg: cl_h_plasma,
        cl_r_ml_min_kg: cl_r_plasma,
        cl_tot_ml_min_kg: cl_tot,
        eh,
        fh,
        vdss_l_kg: vdss_l_kg.clamp(0.08, 60.0),
        peff_1e4,
        sol_mg_ml,
        fg,
        ka_per_h,
        kpuu_brain,
        frac_central,
    }
}

fn compute_oral_absorption(p: &PkSampleParams, dose_mg: f64, hia_p: f64) -> (f64, f64, f64) {
    let an = p.peff_1e4 * 1.366; // Peff * 2 * t_si(3.32h) / R(1.75cm)
    let fa_perm = (1.0 - (-2.0 * an).exp()).clamp(0.05, 0.995);
    // Standard FDA BCS static Dose Number (250 mL FaSSIF)
    let do_num = (dose_mg.max(0.01) / (250.0 * p.sol_mg_ml.max(1.0e-5))).min(1.0e4);
    // Butler & Dressman (2010) DCS dynamic intestinal sink volume (500 mL for high-perm BCS IIa)
    let v_sink_ml = if p.peff_1e4 >= 1.5 { 500.0 } else { 250.0 };
    let do_eff = (dose_mg.max(0.01) / (v_sink_ml * p.sol_mg_ml.max(1.0e-5))).min(1.0e4);
    let fa_diss = if do_eff <= 1.0 {
        1.0
    } else {
        (1.0 / do_eff.powf(0.50)).clamp(0.08, 1.0)
    };
    let fa_mech = (fa_perm * fa_diss).clamp(0.05, 0.99);
    // Gentle regularization with TDC HIA probability
    let fa = (0.80 * fa_mech + 0.20 * hia_p.clamp(0.1, 0.99) * fa_diss).clamp(0.05, 0.99);
    let f_oral = (fa * p.fg * p.fh).clamp(0.02, 0.98);
    (fa, f_oral, do_num)
}

#[derive(Debug, Clone)]
struct OdeRunOutput {
    cmax_u_nm: f64,
    cmin_u_nm: f64,
    cavg_u_nm: f64,
    auc_0_24_u_nm_h: f64,
    tmax_h: f64,
    accumulation_ratio: f64,
    steady_state_48h: Vec<(f64, f64)>, // (t_h in 0..=48, cu_nm)
}

/// Integrates the mass-conserving 2-compartment + absorption depot Minimal-PBPK ODE via RK4.
fn simulate_rk4_steady_state(
    p: &PkSampleParams,
    mw: f64,
    dose_mg: f64,
    tau_h: f64,
    bioavail_f: f64,
    cfg: &PkSimulationConfig,
) -> OdeRunOutput {
    let bw = cfg.body_weight_kg.clamp(10.0, 200.0);
    let cl_l_h = p.cl_tot_ml_min_kg * bw * 60.0 / 1000.0;
    let vdss_l = p.vdss_l_kg * bw;
    let vc = (vdss_l * p.frac_central).max(3.0);
    let vp = (vdss_l - vc).max(0.5);
    let q12 = (0.35 * vc).clamp(2.0, 40.0);

    let t_half_h = (std::f64::consts::LN_2 * vdss_l) / cl_l_h.max(1.0e-4);
    let sim_days = (5.0 * t_half_h / 24.0).ceil().clamp(7.0, 28.0) as usize;
    let total_h = (sim_days as f64) * 24.0;
    let dt = 0.05_f64;
    let n_steps = (total_h / dt).round() as usize;
    let steps_per_tau = (tau_h / dt).round().max(1.0) as usize;

    let dose_nmol = (dose_mg * 1.0e6) / mw.max(50.0);
    let (ka, f_eff) = match cfg.route {
        PkRoute::Oral => (p.ka_per_h, bioavail_f),
        PkRoute::Iv => (0.0, 1.0),
        PkRoute::Sc => (0.12, 0.75),
    };

    let ke = cl_l_h / vc;
    let k12 = q12 / vc;
    let k21 = q12 / vp;
    let tmdd_vmax_nmol_h = cfg.tmdd_vmax_nm_h.map(|v| v * vc);
    let tmdd_km_nm = cfg.tmdd_km_nm.unwrap_or(10.0).max(1.0e-3);

    let deriv = |ad: f64, ac: f64, ap: f64| -> (f64, f64, f64) {
        let cu_nm = (ac / vc).max(0.0) * p.fup;
        let tmdd_rate = tmdd_vmax_nmol_h.map_or(0.0, |vm| vm * cu_nm / (tmdd_km_nm + cu_nm));
        let dad = -ka * ad;
        let dac = ka * ad - ke * ac - tmdd_rate - k12 * ac + k21 * ap;
        let dap = k12 * ac - k21 * ap;
        (dad, dac, dap)
    };

    let mut ad = 0.0_f64;
    let mut ac = 0.0_f64;
    let mut ap = 0.0_f64;

    // Track first-dose Cmax for accumulation ratio, and final 48h window for steady-state metrics
    let mut cmax_first = 0.0_f64;
    let ss_start_h = (total_h - 48.0).max(0.0);
    let last_tau_start_h = (total_h - tau_h).max(0.0);
    let mut cmax_ss = 0.0_f64;
    let mut cmin_ss = f64::INFINITY;
    let mut tmax_rel = 0.0_f64;
    let mut auc_last_tau = 0.0_f64;
    let mut prev_cu_tau: Option<f64> = None;
    let mut curve_48h = Vec::new();
    let sample_every = (0.5 / dt).round().max(1.0) as usize; // every 0.5 h

    for step in 0..=n_steps {
        let t_h = (step as f64) * dt;
        // Apply bolus/oral/SC dose at start of each dosing interval
        if step < n_steps && step % steps_per_tau == 0 {
            if cfg.route == PkRoute::Iv {
                ac += dose_nmol;
            } else {
                ad += dose_nmol * f_eff;
            }
        }

        let cu_nm = (ac / vc).max(0.0) * p.fup;
        if t_h <= tau_h && cu_nm > cmax_first {
            cmax_first = cu_nm;
        }
        if t_h >= ss_start_h - 1e-9 && (step % sample_every == 0 || step == n_steps) {
            curve_48h.push(((t_h - ss_start_h).clamp(0.0, 48.0), cu_nm));
        }
        if t_h >= last_tau_start_h - 1e-9 {
            if cu_nm > cmax_ss {
                cmax_ss = cu_nm;
                tmax_rel = (t_h - last_tau_start_h).clamp(0.0, tau_h);
            }
            if cu_nm < cmin_ss {
                cmin_ss = cu_nm;
            }
            if let Some(prev) = prev_cu_tau {
                auc_last_tau += 0.5 * (prev + cu_nm) * dt;
            }
            prev_cu_tau = Some(cu_nm);
        }

        if step == n_steps {
            break;
        }
        // Standard 4th-order Runge-Kutta step
        let (k1d, k1c, k1p) = deriv(ad, ac, ap);
        let (k2d, k2c, k2p) = deriv(
            ad + 0.5 * dt * k1d,
            ac + 0.5 * dt * k1c,
            ap + 0.5 * dt * k1p,
        );
        let (k3d, k3c, k3p) = deriv(
            ad + 0.5 * dt * k2d,
            ac + 0.5 * dt * k2c,
            ap + 0.5 * dt * k2p,
        );
        let (k4d, k4c, k4p) = deriv(ad + dt * k3d, ac + dt * k3c, ap + dt * k3p);
        ad = (ad + (dt / 6.0) * (k1d + 2.0 * k2d + 2.0 * k3d + k4d)).max(0.0);
        ac = (ac + (dt / 6.0) * (k1c + 2.0 * k2c + 2.0 * k3c + k4c)).max(0.0);
        ap = (ap + (dt / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)).max(0.0);
    }

    if !cmin_ss.is_finite() {
        cmin_ss = 0.0;
    }
    let cavg_ss = auc_last_tau / tau_h.max(1.0);
    let auc_24 = auc_last_tau * (24.0 / tau_h.max(1.0));
    let rac = if cmax_first > 1e-9 {
        (cmax_ss / cmax_first).clamp(1.0, 50.0)
    } else {
        1.0
    };

    OdeRunOutput {
        cmax_u_nm: cmax_ss,
        cmin_u_nm: cmin_ss,
        cavg_u_nm: cavg_ss,
        auc_0_24_u_nm_h: auc_24,
        tmax_h: tmax_rel,
        accumulation_ratio: rac,
        steady_state_48h: curve_48h,
    }
}

fn solve_regimen_dose(
    p: &PkSampleParams,
    mw: f64,
    tau_h: f64,
    hia_p: f64,
    cfg: &PkSimulationConfig,
) -> (f64, f64, OdeRunOutput) {
    let target_cu = (cfg.target_potency_nm * cfg.target_fold).max(0.01);

    if cfg.tmdd_vmax_nm_h.is_some() {
        // Nonlinear PK (TMDD): bracketed log-space bisection
        let mut lo = 0.05_f64;
        let mut hi = 20_000.0_f64;
        for _ in 0..22 {
            let mid = (lo * hi).sqrt();
            let (_, f_oral, _) = compute_oral_absorption(p, mid, hia_p);
            let out = simulate_rk4_steady_state(p, mw, mid, tau_h, f_oral, cfg);
            let obs = if cfg.match_trough {
                out.cmin_u_nm
            } else {
                out.cavg_u_nm
            };
            if obs < target_cu {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let dose_mg = (lo * hi).sqrt();
        let (_, f_oral, do_num) = compute_oral_absorption(p, dose_mg, hia_p);
        let out = simulate_rk4_steady_state(p, mw, dose_mg, tau_h, f_oral, cfg);
        return (dose_mg, do_num, out);
    }

    // Linear PK: run RK4 once at unit dose (F = 1.0 for Oral; built-in route F for IV/SC)
    // and solve exact monotonic dose * F_oral(dose) * unit_metric == target_cu via bisection.
    let unit_out = simulate_rk4_steady_state(p, mw, 1.0, tau_h, 1.0, cfg);
    let unit_metric = if cfg.match_trough {
        unit_out
            .cmin_u_nm
            .max(unit_out.cavg_u_nm * 0.02)
            .max(1.0e-6)
    } else {
        unit_out.cavg_u_nm.max(1.0e-6)
    };

    let dose_mg = if cfg.route == PkRoute::Oral {
        let mut lo = 0.05_f64;
        let mut hi = 25_000.0_f64;
        for _ in 0..40 {
            let mid = (lo * hi).sqrt();
            let (_, f_oral, _) = compute_oral_absorption(p, mid, hia_p);
            if mid * f_oral * unit_metric < target_cu {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (lo * hi).sqrt()
    } else {
        (target_cu / unit_metric).clamp(0.05, 25_000.0)
    };

    let (_, f_oral, do_num) = compute_oral_absorption(p, dose_mg, hia_p);
    let out = simulate_rk4_steady_state(p, mw, dose_mg, tau_h, f_oral, cfg);
    (dose_mg, do_num, out)
}

fn herg_ic50_est_nm(herg_prob: f64) -> f64 {
    // Map calibrated hERG probability (10 uM TDC threshold) to conservative IC50 estimate (nM)
    if herg_prob >= 0.75 {
        1_500.0
    } else if herg_prob >= 0.50 {
        5_000.0
    } else if herg_prob >= 0.30 {
        15_000.0
    } else {
        40_000.0
    }
}

fn classify_feasibility(daily_dose_mg: f64) -> String {
    if daily_dose_mg <= 100.0 {
        "Ideal (<=100 mg/day)".to_string()
    } else if daily_dose_mg <= 500.0 {
        "Feasible (100-500 mg/day)".to_string()
    } else if daily_dose_mg <= 1500.0 {
        "High Dose (500-1500 mg/day)".to_string()
    } else {
        "Infeasible (>1500 mg/day)".to_string()
    }
}

fn next_unit_rand(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Box-Muller standard normal clipped to `[-1.645, +1.645]` (90% conformal interval endpoints).
fn sample_truncated_norm_unit(state: &mut u64) -> f64 {
    let u1 = next_unit_rand(state).max(1.0e-12);
    let u2 = next_unit_rand(state);
    let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
    (z / 1.644_85).clamp(-1.0, 1.0)
}

fn sample_interval(med: f64, lo: f64, hi: f64, z_unit: f64, log_scale: bool) -> f64 {
    if log_scale {
        let lm = med.max(1e-6).ln();
        let ll = lo.max(1e-6).ln();
        let lh = hi.max(1e-6).ln();
        if z_unit >= 0.0 {
            (lm + z_unit * (lh - lm)).exp()
        } else {
            (lm + z_unit * (lm - ll)).exp()
        }
    } else if z_unit >= 0.0 {
        med + z_unit * (hi - med)
    } else {
        med + z_unit * (med - lo)
    }
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p.clamp(0.0, 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Runs the complete IVIVE → Minimal-PBPK human PK and therapeutic dose simulation on `eval`.
#[must_use]
pub fn simulate_human_pk_dose(
    eval: &MoleculeEvaluation,
    cfg: &PkSimulationConfig,
) -> PbpkHumanDoseReport {
    let mw = eval.physicochemical.molecular_weight.max(50.0);
    let BaselineQsarInputs {
        estimated,
        ppbr,
        cl_mic,
        vdss,
        caco2,
        sol_mg_ml,
        t_half_obach,
        cyp3a4: cyp3a4_p,
        pgp: pgp_p,
        bbb: bbb_p,
        hia: hia_p,
        herg: herg_p,
    } = extract_baseline_and_ci90(eval);
    let qsar_pack_loaded = eval.qsar.is_some();

    let base = build_pk_params(
        eval, ppbr.0, cl_mic.0, vdss.0, caco2.0, sol_mg_ml, cyp3a4_p, pgp_p, bbb_p,
    );

    let (qd_dose, qd_do, qd_out) = solve_regimen_dose(&base, mw, 24.0, hia_p, cfg);
    let (bid_dose, bid_do, bid_out) = solve_regimen_dose(&base, mw, 12.0, hia_p, cfg);
    // Pick curve regimen (QD if feasible or lower than 1.35x BID daily, else BID)
    let use_qd_curve = qd_dose <= bid_dose * 2.0 * 1.35 && qd_dose <= 1000.0;
    let curve_tau = if use_qd_curve { 24.0 } else { 12.0 };
    let curve_dose = if use_qd_curve { qd_dose } else { bid_dose };
    let (fa_ref, f_oral_ref, _) = compute_oral_absorption(&base, curve_dose, hia_p);

    // 200-draw deterministic Monte Carlo propagation of the 90% Split Conformal QSAR intervals
    let n_mc = 200usize;
    let mut rng_state = 0x5042_504B_2026_1121_u64;
    let mut mc_qd_doses = Vec::with_capacity(n_mc);
    let mut mc_bid_doses = Vec::with_capacity(n_mc);
    let n_pts = qd_out.steady_state_48h.len();
    let mut mc_curves: Vec<Vec<f64>> = vec![Vec::with_capacity(n_mc); n_pts];

    let base_curve_out = if use_qd_curve { &qd_out } else { &bid_out };
    if mc_curves.len() != base_curve_out.steady_state_48h.len() {
        mc_curves = vec![Vec::with_capacity(n_mc); base_curve_out.steady_state_48h.len()];
    }

    for _ in 0..n_mc {
        let z_ppbr = sample_truncated_norm_unit(&mut rng_state);
        let z_cl = sample_truncated_norm_unit(&mut rng_state);
        let z_vd = sample_truncated_norm_unit(&mut rng_state);
        let z_caco = sample_truncated_norm_unit(&mut rng_state);
        let p_s = build_pk_params(
            eval,
            sample_interval(ppbr.0, ppbr.1, ppbr.2, z_ppbr, false),
            sample_interval(cl_mic.0, cl_mic.1, cl_mic.2, z_cl, true),
            sample_interval(vdss.0, vdss.1, vdss.2, z_vd, true),
            sample_interval(caco2.0, caco2.1, caco2.2, z_caco, false),
            sol_mg_ml,
            cyp3a4_p,
            pgp_p,
            bbb_p,
        );
        let (d_qd, _, _) = solve_regimen_dose(&p_s, mw, 24.0, hia_p, cfg);
        let (d_bid, _, _) = solve_regimen_dose(&p_s, mw, 12.0, hia_p, cfg);
        mc_qd_doses.push(d_qd);
        mc_bid_doses.push(d_bid * 2.0);

        // Simulate fixed recommended dose under uncertain PK parameters for the C_u(t) 90% CI ribbon
        let (_, f_s, _) = compute_oral_absorption(&p_s, curve_dose, hia_p);
        let c_out = simulate_rk4_steady_state(&p_s, mw, curve_dose, curve_tau, f_s, cfg);
        for (slot, &(_, cu)) in mc_curves.iter_mut().zip(c_out.steady_state_48h.iter()) {
            slot.push(cu);
        }
    }
    mc_qd_doses.sort_by(f64::total_cmp);
    mc_bid_doses.sort_by(f64::total_cmp);

    let qd_ci90 = [
        round2(percentile_sorted(&mc_qd_doses, 0.05)),
        round2(percentile_sorted(&mc_qd_doses, 0.95)),
    ];
    let bid_ci90 = [
        round2(percentile_sorted(&mc_bid_doses, 0.05)),
        round2(percentile_sorted(&mc_bid_doses, 0.95)),
    ];

    let mut steady_state_curve = Vec::with_capacity(base_curve_out.steady_state_48h.len());
    for (idx, &(t_h, cu_med)) in base_curve_out.steady_state_48h.iter().enumerate() {
        let mut vals = mc_curves[idx].clone();
        vals.sort_by(f64::total_cmp);
        steady_state_curve.push(PkTimePoint {
            time_h: round2(t_h),
            cu_median_nm: round2(cu_med),
            cu_p5_nm: round2(percentile_sorted(&vals, 0.05)),
            cu_p95_nm: round2(percentile_sorted(&vals, 0.95)),
        });
    }

    let herg_ic50 = herg_ic50_est_nm(herg_p);
    let qd_est = RegimenDoseEstimate {
        regimen_label: "QD (q24h)".to_string(),
        tau_h: 24.0,
        dose_mg_per_admin: round2(qd_dose),
        daily_dose_mg: round2(qd_dose),
        daily_dose_ci90_mg: qd_ci90,
        cmax_u_nm: round2(qd_out.cmax_u_nm),
        cmin_u_nm: round2(qd_out.cmin_u_nm),
        cavg_u_nm: round2(qd_out.cavg_u_nm),
        auc_0_24_u_nm_h: round2(qd_out.auc_0_24_u_nm_h),
        tmax_h: round2(qd_out.tmax_h),
        accumulation_ratio: round2(qd_out.accumulation_ratio),
        cbrain_avg_u_nm: round2(qd_out.cavg_u_nm * base.kpuu_brain),
        dose_number_do: round2(qd_do),
        solubility_limited: qd_do > 1.0 && cfg.route == PkRoute::Oral,
        herg_safety_margin: round1((herg_ic50 / qd_out.cmax_u_nm.max(0.01)).min(9999.0)),
        feasibility: classify_feasibility(qd_dose),
    };
    let bid_daily = bid_dose * 2.0;
    let bid_est = RegimenDoseEstimate {
        regimen_label: "BID (q12h)".to_string(),
        tau_h: 12.0,
        dose_mg_per_admin: round2(bid_dose),
        daily_dose_mg: round2(bid_daily),
        daily_dose_ci90_mg: bid_ci90,
        cmax_u_nm: round2(bid_out.cmax_u_nm),
        cmin_u_nm: round2(bid_out.cmin_u_nm),
        cavg_u_nm: round2(bid_out.cavg_u_nm),
        auc_0_24_u_nm_h: round2(bid_out.auc_0_24_u_nm_h),
        tmax_h: round2(bid_out.tmax_h),
        accumulation_ratio: round2(bid_out.accumulation_ratio),
        cbrain_avg_u_nm: round2(bid_out.cavg_u_nm * base.kpuu_brain),
        dose_number_do: round2(bid_do),
        solubility_limited: bid_do > 1.0 && cfg.route == PkRoute::Oral,
        herg_safety_margin: round1((herg_ic50 / bid_out.cmax_u_nm.max(0.01)).min(9999.0)),
        feasibility: classify_feasibility(bid_daily),
    };

    let t_half_ivive_h =
        (std::f64::consts::LN_2 * base.vdss_l_kg * 1000.0) / (base.cl_tot_ml_min_kg * 60.0);
    let t_half_consensus_h = t_half_obach.map_or(t_half_ivive_h, |ob| (t_half_ivive_h * ob).sqrt());

    let high_sol = qd_do <= 1.0;
    let high_perm = base.peff_1e4 >= 1.5;
    let bcs_class = match (high_sol, high_perm) {
        (true, true) => BcsClass::ClassI,
        (false, true) => BcsClass::ClassII,
        (true, false) => BcsClass::ClassIII,
        (false, false) => BcsClass::ClassIV,
    };

    let recommended_regimen = if use_qd_curve {
        format!(
            "{} mg QD ({}) [90% CI: {:.0}-{:.0} mg/d]",
            round1(qd_est.dose_mg_per_admin),
            cfg.route.as_str(),
            qd_est.daily_dose_ci90_mg[0],
            qd_est.daily_dose_ci90_mg[1]
        )
    } else {
        format!(
            "{} mg BID ({:.0} mg/d, {}) [90% CI: {:.0}-{:.0} mg/d]",
            round1(bid_est.dose_mg_per_admin),
            bid_est.daily_dose_mg,
            cfg.route.as_str(),
            bid_est.daily_dose_ci90_mg[0],
            bid_est.daily_dose_ci90_mg[1]
        )
    };

    let mut warnings = Vec::new();

    // Say it first, before any number is read: with no trained pack behind these inputs the
    // dose and its interval are structural estimates from a logD correlation, not
    // predictions. Silence here is what made the output indistinguishable from a real one.
    if !qsar_pack_loaded {
        warnings.push(
            "No QSAR model pack is loaded: every ADMET input below is a logD-based \
             correlation, not a trained prediction. The dose and its 90% interval are \
             structural estimates — install the model pack before using them."
                .to_string(),
        );
    } else if !estimated.is_empty() {
        warnings.push(format!(
            "No QSAR model pack entry for {}: {} input(s) fell back to a logD-based \
             correlation rather than a trained prediction.",
            estimated.join(", "),
            estimated.len()
        ));
    }

    if let Some(ob) = t_half_obach {
        let ratio = (t_half_ivive_h / ob.max(0.1)).max(ob / t_half_ivive_h.max(0.1));
        if ratio > 3.0 {
            warnings.push(format!(
                "IVIVE t1/2 ({t_half_ivive_h:.1} h) diverges {ratio:.1}x from Obach QSAR t1/2 ({ob:.1} h): verify non-CYP clearance or active renal/biliary transport."
            ));
        }
    }
    if qd_est.solubility_limited {
        warnings.push(format!(
            "Dissolution/solubility-limited oral absorption at QD dose (Do = {:.1} > 1.0; formulation enabling or salt/ASD recommended).",
            qd_est.dose_number_do
        ));
    }
    if qd_est.herg_safety_margin < 30.0 {
        warnings.push(format!(
            "Narrow hERG safety margin at QD Cmax,u ({:.1}x < 30x target; consider BID dosing or lowering LogD/basicity).",
            qd_est.herg_safety_margin
        ));
    }
    if base.eh > 0.65 && cfg.route == PkRoute::Oral {
        warnings.push(format!(
            "High hepatic first-pass extraction (E_H = {:.2}, F_H = {:.0}%): metabolic SoM blocking recommended.",
            base.eh,
            base.fh * 100.0
        ));
    }

    PbpkHumanDoseReport {
        molecule_name: eval.name.clone(),
        canonical_smiles: eval.canonical_smiles.clone(),
        molecular_weight: round2(mw),
        route: cfg.route,
        bcs_class,
        fup: round4(base.fup),
        fu_inc: round4(base.fu_inc),
        blood_to_plasma_rb: round2(base.rb),
        cl_int_u_ul_min_mg: round2(base.cl_int_u),
        cl_hepatic_ml_min_kg: round2(base.cl_h_ml_min_kg),
        cl_renal_ml_min_kg: round2(base.cl_r_ml_min_kg),
        cl_total_ml_min_kg: round2(base.cl_tot_ml_min_kg),
        hepatic_extraction_eh: round3(base.eh),
        vdss_l_kg: round2(base.vdss_l_kg),
        t_half_ivive_h: round2(t_half_ivive_h),
        t_half_obach_h: t_half_obach.map(round2),
        t_half_consensus_h: round2(t_half_consensus_h),
        peff_1e4_cm_s: round2(base.peff_1e4),
        fa: round3(fa_ref),
        fg: round3(base.fg),
        fh: round3(base.fh),
        oral_bioavailability_f: round3(f_oral_ref),
        ka_per_h: round2(base.ka_per_h),
        kpuu_brain: round3(base.kpuu_brain),
        target_potency_nm: round2(cfg.target_potency_nm),
        target_fold: round2(cfg.target_fold),
        match_trough: cfg.match_trough,
        qd: qd_est,
        bid: bid_est,
        recommended_regimen,
        steady_state_curve,
        warnings,
        qsar_pack_loaded,
        estimated_inputs: estimated,
    }
}

/// Renders a clean publication-ready Markdown summary for a human PK/PBPK dose report.
#[must_use]
pub fn render_pk_dose_markdown(r: &PbpkHumanDoseReport) -> String {
    let mut md = String::new();
    let _ = writeln!(
        md,
        "# Human PK / Minimal-PBPK Therapeutic Dose Report: {}\n",
        r.molecule_name
    );
    // Before any number: a reader must not have to reach the warnings section to learn
    // that nothing here came out of a trained model.
    if !r.qsar_pack_loaded {
        let _ = writeln!(
            md,
            "> **No QSAR model pack loaded.** Every ADMET input below is a logD-based \
             correlation, not a trained prediction, and the dose and its 90% interval are \
             structural estimates. Install the model pack before using these numbers.\n"
        );
    }
    let _ = writeln!(
        md,
        "- **SMILES:** `{}` | **MW:** `{:.2} Da` | **Route:** `{}` | **BCS:** `{}`",
        r.canonical_smiles,
        r.molecular_weight,
        r.route.as_str(),
        r.bcs_class.as_str()
    );
    let criterion = if r.match_trough {
        format!(
            "Cmin,u >= {:.1}x IC50 ({:.1} nM)",
            r.target_fold, r.target_potency_nm
        )
    } else {
        format!(
            "Cavg,u >= {:.1}x IC50 ({:.1} nM)",
            r.target_fold, r.target_potency_nm
        )
    };
    let _ = writeln!(
        md,
        "- **Target Coverage:** `{criterion}` | **Recommended Regimen:** **{}**\n",
        r.recommended_regimen
    );

    md.push_str("## 1. Mechanistic Human IVIVE & Absorption Parameters (70 kg Adult)\n\n");
    md.push_str("| Parameter | Value | Parameter | Value |\n|---|---:|---|---:|\n");
    let _ = writeln!(
        md,
        "| Plasma Unbound Fraction (`f_u,p`) | {:.4} | Oral Bioavailability (`F = Fa·Fg·Fh`) | **{:.1}%** ({:.2}·{:.2}·{:.2}) |",
        r.fup,
        r.oral_bioavailability_f * 100.0,
        r.fa,
        r.fg,
        r.fh
    );
    let _ = writeln!(
        md,
        "| Hepatic Clearance (`CL_H`) | {:.2} mL/min/kg (`E_H`={:.2}) | Jejunal `P_eff` / `k_a` | {:.2}×10⁻⁴ cm/s / {:.2} h⁻¹ |",
        r.cl_hepatic_ml_min_kg, r.hepatic_extraction_eh, r.peff_1e4_cm_s, r.ka_per_h
    );
    let _ = writeln!(
        md,
        "| Total Clearance (`CL_tot`) | **{:.2} mL/min/kg** (`CL_R`={:.2}) | Volume of Distribution (`V_d,ss`) | **{:.2} L/kg** |",
        r.cl_total_ml_min_kg, r.cl_renal_ml_min_kg, r.vdss_l_kg
    );
    let ob_s = r
        .t_half_obach_h
        .map_or_else(|| "—".to_string(), |v| format!("{v:.1} h"));
    let _ = writeln!(
        md,
        "| Effective Half-Life (`t1/2` IVIVE / Obach) | **{:.1} h** / {} | CNS Partition (`K_p,uu,brain`) | {:.3} |\n",
        r.t_half_ivive_h, ob_s, r.kpuu_brain
    );

    md.push_str("## 2. Predicted Human Therapeutic Dose Regimens (with 90% Conformal CI)\n\n");
    md.push_str(
        "| Regimen | Dose / Admin | Daily Dose (90% CI) | Cmax,u (nM) | Cmin,u (nM) | AUC_0-24,u | Do | hERG Margin | Feasibility |\n|---|---:|---|---:|---:|---:|---:|---:|---|\n",
    );
    for reg in [&r.qd, &r.bid] {
        let _ = writeln!(
            md,
            "| **{}** | **{:.1} mg** | {:.1} mg ({:.0}-{:.0}) | {:.1} | {:.1} | {:.0} | {:.2} | {:.0}× | {} |",
            reg.regimen_label,
            reg.dose_mg_per_admin,
            reg.daily_dose_mg,
            reg.daily_dose_ci90_mg[0],
            reg.daily_dose_ci90_mg[1],
            reg.cmax_u_nm,
            reg.cmin_u_nm,
            reg.auc_0_24_u_nm_h,
            reg.dose_number_do,
            reg.herg_safety_margin,
            reg.feasibility
        );
    }
    if !r.warnings.is_empty() {
        md.push_str("\n## 3. Translational DMPK Alerts\n\n");
        for w in &r.warnings {
            let _ = writeln!(md, "- {w}");
        }
    }
    md
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}
fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}
fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}
fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluate_smiles_full;
    use admet_calc::TppPreset;

    /// Audit findings F3-14 and F4a-07: with no QSAR model pack installed, every endpoint
    /// falls back to a correlation on logD and the report still reads like a measured
    /// prediction — a human dose with a "90% CI" built from fixed multipliers. The caller
    /// has no way to tell. `model_info` does say there is no pack; this report did not.
    #[test]
    fn a_dose_computed_without_a_model_pack_says_so() -> Result<(), Box<dyn std::error::Error>> {
        // evaluate_smiles_full with no pack path: eval.qsar stays None.
        let eval = evaluate_smiles_full(
            "CC(=O)Oc1ccccc1C(=O)O",
            Some("Aspirin"),
            None,
            TppPreset::OralSmallMolecule,
        )?;
        assert!(
            eval.qsar.is_none(),
            "this test needs the no-pack path to be the one under test"
        );

        let report = simulate_human_pk_dose(&eval, &PkSimulationConfig::default());

        assert!(
            !report.qsar_pack_loaded,
            "the report must record that no QSAR pack backed these numbers"
        );
        assert!(
            !report.estimated_inputs.is_empty(),
            "the endpoints that fell back to a correlation must be named"
        );
        assert!(
            report
                .warnings
                .iter()
                .any(|w| w.to_ascii_lowercase().contains("no qsar model pack")),
            "a reader of the warnings must be told, got: {:?}",
            report.warnings
        );
        Ok(())
    }

    #[test]
    fn rk4_iv_bolus_conserves_mass_to_analytical_auc_within_2_pct()
    -> Result<(), Box<dyn std::error::Error>> {
        let eval = evaluate_smiles_full(
            "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
            Some("Ibuprofen"),
            None,
            TppPreset::OralSmallMolecule,
        )?;
        let cfg = PkSimulationConfig {
            target_potency_nm: 50.0,
            target_fold: 1.0,
            match_trough: false,
            route: PkRoute::Iv,
            body_weight_kg: 70.0,
            ..Default::default()
        };
        let rep = simulate_human_pk_dose(&eval, &cfg);
        // Analytical steady-state unbound AUC_0-24 for IV dosing: (Dose_nmol * fup) / CL_L_h
        let dose_nmol = rep.qd.dose_mg_per_admin * 1.0e6 / rep.molecular_weight;
        let cl_l_h = rep.cl_total_ml_min_kg * 70.0 * 60.0 / 1000.0;
        let expected_auc_u = (dose_nmol * rep.fup) / cl_l_h;
        let rel_err = (rep.qd.auc_0_24_u_nm_h - expected_auc_u).abs() / expected_auc_u.max(1.0);
        assert!(
            rel_err < 0.03,
            "RK4 mass balance error {:.3}% (got {:.1}, expected {:.1})",
            rel_err * 100.0,
            rep.qd.auc_0_24_u_nm_h,
            expected_auc_u
        );
        assert!(rep.qd.daily_dose_ci90_mg[0] <= rep.qd.daily_dose_mg + 1.0);
        assert!(rep.qd.daily_dose_ci90_mg[1] >= rep.qd.daily_dose_mg - 1.0);
        assert!(!rep.steady_state_curve.is_empty());
        Ok(())
    }

    #[test]
    fn oral_sc_and_tmdd_modes_produce_valid_monotonic_human_doses()
    -> Result<(), Box<dyn std::error::Error>> {
        let eval = evaluate_smiles_full(
            "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
            Some("Imatinib"),
            None,
            TppPreset::OncologyKinase,
        )?;
        let rep_10nm = simulate_human_pk_dose(
            &eval,
            &PkSimulationConfig {
                target_potency_nm: 10.0,
                route: PkRoute::Oral,
                ..Default::default()
            },
        );
        let rep_100nm = simulate_human_pk_dose(
            &eval,
            &PkSimulationConfig {
                target_potency_nm: 100.0,
                route: PkRoute::Oral,
                ..Default::default()
            },
        );
        assert!(rep_100nm.qd.daily_dose_mg > rep_10nm.qd.daily_dose_mg * 5.0);
        assert!(rep_10nm.bid.daily_dose_mg <= rep_10nm.qd.daily_dose_mg + 5.0);

        let rep_sc_tmdd = simulate_human_pk_dose(
            &eval,
            &PkSimulationConfig {
                target_potency_nm: 20.0,
                route: PkRoute::Sc,
                tmdd_vmax_nm_h: Some(5.0),
                tmdd_km_nm: Some(10.0),
                ..Default::default()
            },
        );
        assert!(rep_sc_tmdd.qd.daily_dose_mg > 0.0 && rep_sc_tmdd.qd.cmin_u_nm >= 15.0);
        let md = render_pk_dose_markdown(&rep_10nm);
        assert!(md.contains("Human PK / Minimal-PBPK") && md.contains("QD (q24h)"));
        Ok(())
    }
}
