use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

/// Gastrointestinal passive absorption class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AbsorptionLevel {
    High,
    Low,
}

impl AbsorptionLevel {
    /// Display label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::High => "High",
            Self::Low => "Low",
        }
    }
}

/// Qualitative aqueous solubility class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolubilityClass {
    Insoluble,
    PoorlySoluble,
    ModeratelySoluble,
    Soluble,
    VerySoluble,
    HighlySoluble,
}

impl SolubilityClass {
    /// Classify a `log S` (mol/L) value.
    pub fn from_log_s(log_s: f64) -> Self {
        if log_s < -10.0 {
            Self::Insoluble
        } else if log_s < -6.0 {
            Self::PoorlySoluble
        } else if log_s < -4.0 {
            Self::ModeratelySoluble
        } else if log_s < -2.0 {
            Self::Soluble
        } else if log_s < 0.0 {
            Self::VerySoluble
        } else {
            Self::HighlySoluble
        }
    }

    /// Display label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Insoluble => "Insoluble",
            Self::PoorlySoluble => "Poorly soluble",
            Self::ModeratelySoluble => "Moderately soluble",
            Self::Soluble => "Soluble",
            Self::VerySoluble => "Very soluble",
            Self::HighlySoluble => "Highly soluble",
        }
    }
}

/// Physicochemical descriptors section.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicochemicalProperties {
    pub formula: String,
    pub molecular_weight: f64,
    #[serde(default)]
    pub exact_mass: f64,
    pub heavy_atoms: usize,
    pub aromatic_heavy_atoms: usize,
    pub fraction_csp3: f64,
    pub rotatable_bonds: usize,
    pub h_bond_acceptors: usize,
    pub h_bond_donors: usize,
    pub molar_refractivity: f64,
    /// Ertl (2000) TPSA including N, O, S, and P (SwissADME convention).
    pub tpsa: f64,
    /// Ertl (2000) TPSA counting N and O only (RDKit / QED convention).
    #[serde(default)]
    pub tpsa_no_only: f64,
    /// Double Bond Equivalents (Rings + Double Bonds).
    #[serde(default)]
    pub dbe: f64,
    #[serde(default)]
    pub num_rings: usize,
    #[serde(default)]
    pub num_aromatic_rings: usize,
}

/// Dominant physiological ionization state at pH 7.4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum IonizationClass {
    #[default]
    Neutral,
    Acid,
    Base,
    Zwitterion,
}

impl IonizationClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Neutral => "Neutral",
            Self::Acid => "Acid",
            Self::Base => "Base",
            Self::Zwitterion => "Zwitterion",
        }
    }
}

/// Physiological pKa, ionization class, and Henderson-Hasselbalch LogD at pH 7.4.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct IonizationProfile {
    pub strongest_acidic_pka: Option<f64>,
    pub strongest_basic_pka: Option<f64>,
    pub ionization_class: IonizationClass,
    pub estimated_charge_ph74: f64,
    pub logd_ph74: f64,
}

/// Qualitative hERG cardiotoxicity structural/physicochemical risk class (heuristic).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum HergRisk {
    #[default]
    Low,
    Moderate,
    High,
}

impl HergRisk {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Moderate => "Moderate",
            Self::High => "High",
        }
    }
}

/// Extended toxicity & reactive-metabolite structural safety screening panel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SafetyProfile {
    pub herg_risk: HergRisk,
    pub herg_reasons: Vec<String>,
    pub ames_alerts: Vec<String>,
    pub dili_alerts: Vec<String>,
    pub safety_alerts_are_structural_heuristics: bool,
}

/// Lipophilicity predictions: 3 distinct models (`XLOGP3`-heuristic, `WLOGP`, `MLOGP`),
/// their arithmetic mean (`consensus_log_p`), plus 2 clearly labeled experimental linear surrogates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lipophilicity {
    /// Wildman & Crippen (1999) atomic logP.
    pub wlogp: f64,
    /// Atom-additive logP heuristic inspired by Cheng et al. (2007).
    pub xlogp3: f64,
    /// Moriguchi et al. (1992) topological logP.
    pub mlogp: f64,
    /// Experimental solvation-inspired linear heuristic (NOT SwissADME iLOGP).
    pub solvation_logp_est: f64,
    /// Experimental topological linear heuristic (NOT Silicos-IT).
    pub topological_logp_est: f64,
    /// Arithmetic mean of the 3 primary models (`wlogp`, `xlogp3`, `mlogp`).
    pub consensus_log_p: f64,
}

/// Single water solubility model output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaterSolubilityModel {
    pub log_s: f64,
    pub solubility_mg_ml: f64,
    pub solubility_mol_l: f64,
    pub class: SolubilityClass,
}

/// Water solubility section (ESOL, Ali, and an experimental topological heuristic).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaterSolubility {
    pub esol: WaterSolubilityModel,
    pub ali: WaterSolubilityModel,
    pub topological_est: WaterSolubilityModel,
}

/// Pharmacokinetics (ADME) predictions.
/// Note: GI/BBB use the BOILED-Egg ellipses and log Kp uses Potts & Guy (1992);
/// P-gp substrate and CYP450 flags are experimental rule-based heuristics (not trained SVMs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pharmacokinetics {
    pub gi_absorption: AbsorptionLevel,
    pub bbb_permeant: bool,
    /// Experimental heuristic P-gp substrate flag.
    pub pgp_substrate: bool,
    /// Experimental heuristic CYP1A2 inhibitor flag.
    pub cyp1a2_inhibitor: bool,
    /// Experimental heuristic CYP2C19 inhibitor flag.
    pub cyp2c19_inhibitor: bool,
    /// Experimental heuristic CYP2C9 inhibitor flag.
    pub cyp2c9_inhibitor: bool,
    /// Experimental heuristic CYP2D6 inhibitor flag.
    pub cyp2d6_inhibitor: bool,
    /// Experimental heuristic CYP3A4 inhibitor flag.
    pub cyp3a4_inhibitor: bool,
    pub log_kp_cm_s: f64,
    pub pgp_cyp_are_experimental_heuristics: bool,
}

/// Outcome of a druglikeness / leadlikeness rule filter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RuleResult {
    pub pass: bool,
    pub violations: usize,
    pub details: Vec<String>,
}

/// Druglikeness section (SwissADME 5 filters + Bickerton QED + Pfizer CNS MPO + Industrial rules).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Druglikeness {
    pub lipinski: RuleResult,
    pub ghose: RuleResult,
    pub veber: RuleResult,
    pub egan: RuleResult,
    pub muegge: RuleResult,
    pub bioavailability_score: f64,
    /// Bickerton et al. (2012, *Nature Chemistry*) Quantitative Estimate of Drug-likeness `[0.0, 1.0]`.
    #[serde(default)]
    pub qed: f64,
    /// Wager et al. (2010) Pfizer CNS Multiparameter Optimization score `[0.0, 6.0]`.
    #[serde(default)]
    pub cns_mpo: f64,
    /// Hughes et al. (2008) Pfizer 3/75 in vivo toxicity rule (`pass = !(LogP > 3 && TPSA < 75)`).
    #[serde(default)]
    pub pfizer_3_75: RuleResult,
    /// Gleeson (2008) GSK 4/400 favorable ADMET space (`MW <= 400 && LogP <= 4`).
    #[serde(default)]
    pub gsk_4_400: RuleResult,
    /// Johnson et al. (2009) Golden Triangle (`200 <= MW <= 500 && -2 <= LogD7.4 <= 5`).
    #[serde(default)]
    pub golden_triangle: RuleResult,
    /// Congreve et al. (2003) Fragment Rule of Three (`MW <= 300, LogP <= 3, HBD <= 3, HBA <= 3, RB <= 3`).
    #[serde(default)]
    pub ro3_fragment: RuleResult,
    /// Doak et al. (2014) Beyond Rule of 5 (`MW > 500` or `TPSA > 140` or `HBD > 5` or `RB > 10`).
    #[serde(default)]
    pub beyond_ro5: bool,
}

/// Medicinal Chemistry alerts and scores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MedicinalChemistry {
    pub pains_alerts: Vec<String>,
    /// Number of curated PAINS SMARTS checked (subset of Baell & Holloway ~480 catalog).
    pub pains_catalog_size: usize,
    pub brenk_alerts: Vec<String>,
    /// Number of curated Brenk SMARTS checked (subset of Brenk et al. ~105 catalog).
    pub brenk_catalog_size: usize,
    pub leadlikeness: RuleResult,
    /// Heuristic Synthetic Accessibility proxy in `[1.0, 10.0]` (does not include PubChem fragment table).
    pub synthetic_accessibility: f64,
    pub sa_is_heuristic_proxy: bool,
}

/// BOILED-Egg coordinate and region classification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoiledEggPoint {
    pub tpsa: f64,
    pub wlogp: f64,
    pub in_white: bool,
    pub in_yolk: bool,
    pub pgp_substrate: bool,
    pub in_plot_bounds: bool,
}

/// Shared geometry sampler for BOILED-Egg ellipses across PDF, SVG, and GUI renderers.
pub struct BoiledEggGeometry;

impl BoiledEggGeometry {
    /// Sample points in `(tpsa, wlogp)` coordinates along a rotated ellipse.
    pub fn sample_ellipse(
        center: (f64, f64),
        axes: (f64, f64),
        angle_rad: f64,
        steps: usize,
    ) -> Vec<(f64, f64)> {
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();
        (0..steps)
            .map(|i| {
                let t = 2.0 * PI * (i as f64) / (steps as f64);
                let lx = axes.0 * t.cos();
                let ly = axes.1 * t.sin();
                (
                    center.0 + lx * cos_a - ly * sin_a,
                    center.1 + lx * sin_a + ly * cos_a,
                )
            })
            .collect()
    }
}

/// Six-axis Bioavailability Radar descriptors and normalized values `[0, 1]` where `[0.25, 0.85]`
/// highlights the optimal pink physicochemical zone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BioavailabilityRadar {
    pub lipo_xlogp3: f64,
    pub size_mw: f64,
    pub polar_tpsa: f64,
    pub insolu_log_s: f64,
    pub insatu_fsp3: f64,
    pub flex_rot_bonds: usize,
    /// Normalized axis scores `[0.0, 1.0]` in order `[LIPO, SIZE, POLAR, INSOLU, INSATU, FLEX]`.
    pub normalized_axes: [f64; 6],
    /// True if all 6 axes lie inside the optimal physicochemical zone.
    pub all_in_optimal_zone: bool,
}

impl BioavailabilityRadar {
    /// Axis labels in clockwise order starting at 12 o'clock.
    pub const AXIS_LABELS: [&'static str; 6] =
        ["LIPO", "SIZE", "POLAR", "INSOLU", "INSATU", "FLEX"];

    /// Compute normalized unit-circle `(x, y)` coordinates (`x` right, `y` up) for a hexagon at `radius`.
    pub fn unit_hexagon(radius: f64) -> [(f64, f64); 6] {
        let mut pts = [(0.0, 0.0); 6];
        for (i, slot) in pts.iter_mut().enumerate() {
            let ang = PI / 2.0 - (i as f64) * PI / 3.0;
            *slot = (radius * ang.cos(), radius * ang.sin());
        }
        pts
    }

    /// Compute normalized `(x, y)` coordinates (`x` right, `y` up) for this molecule's radar polygon.
    pub fn unit_polygon(&self) -> [(f64, f64); 6] {
        let mut pts = [(0.0, 0.0); 6];
        for (i, (&val, slot)) in self.normalized_axes.iter().zip(pts.iter_mut()).enumerate() {
            let ang = PI / 2.0 - (i as f64) * PI / 3.0;
            let r = val.clamp(0.05, 1.0);
            *slot = (r * ang.cos(), r * ang.sin());
        }
        pts
    }
}

/// Applicability Domain status based on ECFP4 Tanimoto similarity to the training set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ApplicabilityDomainStatus {
    #[default]
    InDomain,
    Borderline,
    OutOfDomain,
}

impl ApplicabilityDomainStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InDomain => "In-Domain",
            Self::Borderline => "Borderline",
            Self::OutOfDomain => "Out-of-Domain",
        }
    }
}

/// Quantitative Applicability Domain metrics against the endpoint training distribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ApplicabilityDomain {
    pub status: ApplicabilityDomainStatus,
    /// Nearest-neighbor ECFP4 Tanimoto similarity to the training set `[0.0, 1.0]`.
    pub max_tanimoto: f64,
    /// Mean Tanimoto similarity to the top-5 nearest training neighbors `[0.0, 1.0]`.
    pub top5_tanimoto: f64,
    /// Standardized physicochemical leverage (RMS Z-score across core descriptors).
    pub descriptor_z_rms: f64,
}

/// Kind of QSAR endpoint prediction (binary classification probability or continuous regression).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QsarTaskKind {
    BinaryClassification,
    Regression,
}

/// Single quantitative TDC QSAR endpoint prediction with 90% Split Conformal uncertainty and AD.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QsarEndpointPrediction {
    pub endpoint_id: String,
    pub display_name: String,
    pub category: String,
    pub task: QsarTaskKind,
    /// Predicted probability `[0, 1]` for classification, or point value for regression.
    pub value: f64,
    /// Lower bound of the 90% split-conformal interval (for regression) or probability lower band.
    pub ci90_low: f64,
    /// Upper bound of the 90% split-conformal interval (for regression) or probability upper band.
    pub ci90_high: f64,
    /// Units string (e.g. `"prob"`, `"log(cm/s)"`, `"%"`, `"L/kg"`, `"uL/min/mg"`, `"h"`, `"log(mol/L)"`).
    pub units: String,
    /// Optional back-transformed linear-scale value + 90% CI for log10-modeled DMPK endpoints
    /// (`VDss` in L/kg, `CLint` in uL/min/mg, `Half-Life` in h).
    #[serde(default)]
    pub linear_value: Option<f64>,
    #[serde(default)]
    pub linear_ci90: Option<[f64; 2]>,
    /// Human-readable conformal call (`"Positive"`, `"Negative"`, `"Uncertain (Both)"` for classification,
    /// or range band for regression).
    pub conformal_call: String,
    /// Local applicability domain assessment for this endpoint.
    pub applicability_domain: ApplicabilityDomain,
    /// Honest held-out Bemis-Murcko scaffold-split test metric (`ROC AUC` for classification,
    /// `Spearman rho` for regression) measured during local training.
    pub test_metric_name: String,
    pub test_metric_value: f64,
    /// Empirical 90% conformal coverage observed on the held-out scaffold test fold.
    pub test_conformal_coverage: f64,
}

/// Complete locally-evaluated TDC QSAR ADMET panel with global Applicability Domain summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct QsarAdmetReport {
    pub model_pack_version: String,
    pub model_pack_source: String,
    pub overall_ad: ApplicabilityDomain,
    pub endpoints: Vec<QsarEndpointPrediction>,
}

/// Risk tier for an atomic Site of Metabolism (SoM).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MetabolismRisk {
    #[default]
    Low,
    Moderate,
    High,
}

impl MetabolismRisk {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Moderate => "Moderate",
            Self::High => "High",
        }
    }
}

/// Predicted atomic Site of Metabolism (CYP3A4 / 2D6 / 2C9 / 1A2 / Phase II UGT & Esterase).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetabolismSite {
    pub atom_index: usize,
    pub atom_symbol: String,
    pub isoform: String,
    pub reaction: String,
    /// Normalized metabolic lability score `[0.0, 1.0]`.
    pub lability_score: f64,
    pub risk: MetabolismRisk,
    pub medchem_fix: String,
}

/// Configurable Target Product Profile (TPP) preset for multiparameter optimization (TPP MPO).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TppPreset {
    #[default]
    OralSmallMolecule,
    CnsPenetrant,
    PeripheralSelective,
    OncologyKinase,
}

impl TppPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OralSmallMolecule => "Oral Small Molecule",
            Self::CnsPenetrant => "CNS Penetrant",
            Self::PeripheralSelective => "Peripheral Selective",
            Self::OncologyKinase => "Oncology / Kinase",
        }
    }

    pub fn slug(self) -> &'static str {
        match self {
            Self::OralSmallMolecule => "oral",
            Self::CnsPenetrant => "cns",
            Self::PeripheralSelective => "peripheral",
            Self::OncologyKinase => "oncology",
        }
    }
}

/// Traffic-light status for an individual TPP objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TppStatus {
    #[default]
    Pass,
    Warn,
    Fail,
}

impl TppStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Warn => "WARN",
            Self::Fail => "FAIL",
        }
    }
}

/// Single objective evaluation inside a Target Product Profile (TPP) report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TppCriterionResult {
    pub name: String,
    pub observed: String,
    pub target_spec: String,
    pub desirability: f64,
    pub weight: f64,
    pub status: TppStatus,
}

/// Multiparameter Target Product Profile (TPP MPO) scorecard & hard safety gates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TppMpoReport {
    pub preset: TppPreset,
    /// Overall weighted desirability score in `[0.0, 1.0]`.
    pub score: f64,
    /// Overall recommendation (`"Advance"`, `"Optimize"`, `"Hold / Red-Flag"`).
    pub decision: String,
    pub hard_stops: Vec<String>,
    pub criteria: Vec<TppCriterionResult>,
}

/// Consolidated summary of Tox21 cellular toxicity & endocrine disruption pathways (12 assays).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Tox21Summary {
    pub assays_evaluated: usize,
    /// Flagged Nuclear Receptor (endocrine/metabolic disruption) assays (`NR-*`).
    pub endocrine_alerts: Vec<String>,
    /// Flagged Cellular Stress Response assays (`SR-*`: oxidative, genotoxic, mitochondrial, p53).
    pub stress_alerts: Vec<String>,
    /// Maximum predicted Tox21 probability across all evaluated assays `[0.0, 1.0]`.
    pub max_probability: f64,
}

/// Risk tier for a predicted off-target liability via SEA (Similarity Ensemble Approach).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OffTargetRisk {
    #[default]
    Low,
    Moderate,
    High,
}

impl OffTargetRisk {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Moderate => "Moderate",
            Self::High => "High",
        }
    }
}

/// Single predicted human off-target hit from pure-Rust SEA ligand-set similarity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OffTargetHit {
    pub target_id: String,
    pub target_name: String,
    pub target_class: String,
    pub organ_liability: String,
    /// Highest ECFP4 Tanimoto similarity to any known active reference ligand of this target.
    pub max_tanimoto: f64,
    /// Name of the closest known active ligand driving this alert.
    pub nearest_ligand: String,
    /// Raw SEA sum of above-threshold Tanimoto similarities.
    pub raw_score: f64,
    /// Background-calibrated SEA Z-score.
    pub z_score: f64,
    /// Analytical extreme-value distribution (Gumbel) E-value (smaller = stronger off-target match).
    pub e_value: f64,
    pub risk: OffTargetRisk,
}

/// Complete local SEA off-target safety pharmacology profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct OffTargetReport {
    pub panel_version: String,
    pub targets_evaluated: usize,
    pub high_risk_count: usize,
    pub moderate_risk_count: usize,
    pub hits: Vec<OffTargetHit>,
}

/// Metabolic biotransformation phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MetabolismPhase {
    #[default]
    PhaseI,
    PhaseII,
}

impl MetabolismPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PhaseI => "Phase I",
            Self::PhaseII => "Phase II",
        }
    }
}

/// Single predicted hepatic/plasma metabolite re-screened for toxicity escalation & bioactivation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictedMetabolite {
    pub rank: usize,
    pub smiles: String,
    pub phase: MetabolismPhase,
    pub reaction: String,
    pub enzyme: String,
    pub parent_som_atom: usize,
    pub lability_score: f64,
    /// True if transformation creates a reactive electrophile / covalent-binding bioactivation species.
    pub reactive_bioactivation: bool,
    pub bioactivation_reason: Option<String>,
    pub dili_prob: Option<f64>,
    pub ames_prob: Option<f64>,
    pub herg_prob: Option<f64>,
    pub max_tox21_prob: Option<f64>,
    /// Structural alerts (PAINS/Brenk/AMES/DILI) present in the metabolite but absent in the parent.
    pub new_alerts: Vec<String>,
    /// True if metabolite introduces reactive bioactivation, new alerts, or significant toxicity jump.
    pub toxicity_escalation: bool,
}

/// Complete Phase I/II Metabolite Guard report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MetaboliteGuardReport {
    pub total_generated: usize,
    pub bioactivation_alerts: usize,
    pub escalation_count: usize,
    pub metabolites: Vec<PredictedMetabolite>,
}

/// Dosing route supported by the mechanistic minimal-PBPK simulator (Oral, IV, SC).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PkRoute {
    #[default]
    Oral,
    Iv,
    Sc,
}

impl PkRoute {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Oral => "Oral",
            Self::Iv => "IV",
            Self::Sc => "SC",
        }
    }
}

/// Biopharmaceutics Classification System (BCS) solubility/permeability class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BcsClass {
    #[default]
    ClassI,
    ClassII,
    ClassIII,
    ClassIV,
}

impl BcsClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ClassI => "BCS Class I (High Sol / High Perm)",
            Self::ClassII => "BCS Class II (Low Sol / High Perm)",
            Self::ClassIII => "BCS Class III (High Sol / Low Perm)",
            Self::ClassIV => "BCS Class IV (Low Sol / Low Perm)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::ClassI => "BCS I",
            Self::ClassII => "BCS II",
            Self::ClassIII => "BCS III",
            Self::ClassIV => "BCS IV",
        }
    }
}

/// Configuration for IVIVE -> Minimal-PBPK human therapeutic dose simulation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PkSimulationConfig {
    /// Target biochemical/cellular potency (`IC50` or `Ki`) in nanomolar (nM).
    pub target_potency_nm: f64,
    /// Required unbound concentration multiple of `target_potency_nm` (default `1.0`).
    pub target_fold: f64,
    /// When `true`, matches unbound trough `Cmin,u >= target_fold * IC50`; when `false`, matches `Cavg,u`.
    pub match_trough: bool,
    pub route: PkRoute,
    pub body_weight_kg: f64,
    /// Optional Michaelis-Menten TMDD maximum saturable clearance (`nM/h`) for future peptide/biologic hooks.
    #[serde(default)]
    pub tmdd_vmax_nm_h: Option<f64>,
    /// Optional Michaelis-Menten TMDD affinity constant (`nM`).
    #[serde(default)]
    pub tmdd_km_nm: Option<f64>,
}

impl Default for PkSimulationConfig {
    fn default() -> Self {
        Self {
            target_potency_nm: 10.0,
            target_fold: 1.0,
            match_trough: true,
            route: PkRoute::Oral,
            body_weight_kg: 70.0,
            tmdd_vmax_nm_h: None,
            tmdd_km_nm: None,
        }
    }
}

/// Single timepoint along the steady-state unbound plasma concentration-time profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PkTimePoint {
    pub time_h: f64,
    pub cu_median_nm: f64,
    pub cu_p5_nm: f64,
    pub cu_p95_nm: f64,
}

/// Predicted human therapeutic dose and steady-state PK metrics for one dosing regimen (QD or BID).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegimenDoseEstimate {
    pub regimen_label: String,
    pub tau_h: f64,
    pub dose_mg_per_admin: f64,
    pub daily_dose_mg: f64,
    pub daily_dose_ci90_mg: [f64; 2],
    pub cmax_u_nm: f64,
    pub cmin_u_nm: f64,
    pub cavg_u_nm: f64,
    pub auc_0_24_u_nm_h: f64,
    pub tmax_h: f64,
    pub accumulation_ratio: f64,
    pub cbrain_avg_u_nm: f64,
    pub dose_number_do: f64,
    pub solubility_limited: bool,
    pub herg_safety_margin: f64,
    pub feasibility: String,
}

/// Complete mechanistic IVIVE -> Minimal-PBPK human PK and therapeutic dose prediction dossier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PbpkHumanDoseReport {
    pub molecule_name: String,
    pub canonical_smiles: String,
    pub molecular_weight: f64,
    pub route: PkRoute,
    pub bcs_class: BcsClass,
    pub fup: f64,
    pub fu_inc: f64,
    pub blood_to_plasma_rb: f64,
    pub cl_int_u_ul_min_mg: f64,
    pub cl_hepatic_ml_min_kg: f64,
    pub cl_renal_ml_min_kg: f64,
    pub cl_total_ml_min_kg: f64,
    pub hepatic_extraction_eh: f64,
    pub vdss_l_kg: f64,
    pub t_half_ivive_h: f64,
    pub t_half_obach_h: Option<f64>,
    pub t_half_consensus_h: f64,
    pub peff_1e4_cm_s: f64,
    pub fa: f64,
    pub fg: f64,
    pub fh: f64,
    pub oral_bioavailability_f: f64,
    pub ka_per_h: f64,
    pub kpuu_brain: f64,
    pub target_potency_nm: f64,
    pub target_fold: f64,
    pub match_trough: bool,
    pub qd: RegimenDoseEstimate,
    pub bid: RegimenDoseEstimate,
    pub recommended_regimen: String,
    pub steady_state_curve: Vec<PkTimePoint>,
    pub warnings: Vec<String>,
}

/// Complete ADMET evaluation for one molecule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoleculeEvaluation {
    pub name: String,
    pub input_smiles: String,
    pub canonical_smiles: String,
    #[serde(default)]
    pub warnings: Vec<String>,
    pub physicochemical: PhysicochemicalProperties,
    pub lipophilicity: Lipophilicity,
    pub water_solubility: WaterSolubility,
    pub pharmacokinetics: Pharmacokinetics,
    pub druglikeness: Druglikeness,
    pub medicinal_chemistry: MedicinalChemistry,
    #[serde(default)]
    pub ionization: IonizationProfile,
    #[serde(default)]
    pub safety: SafetyProfile,
    pub boiled_egg: BoiledEggPoint,
    pub bioavailability_radar: BioavailabilityRadar,
    /// Optional quantitative TDC QSAR panel with 90% Split Conformal intervals & Applicability Domain.
    #[serde(default)]
    pub qsar: Option<QsarAdmetReport>,
    /// Atomic Sites of Metabolism (CYP450 + Phase II UGT / Esterase) sorted by lability.
    #[serde(default)]
    pub metabolism_sites: Vec<MetabolismSite>,
    /// Target Product Profile (TPP) Multiparameter Optimization scorecard.
    #[serde(default)]
    pub tpp_mpo: Option<TppMpoReport>,
    /// Consolidated Tox21 cellular toxicity & nuclear-receptor pathway summary (12 assays).
    #[serde(default)]
    pub tox21_summary: Option<Tox21Summary>,
    /// Local pure-Rust SEA off-target safety pharmacology profile (Safety-20 / ChEMBL panel).
    #[serde(default)]
    pub off_targets: Option<OffTargetReport>,
    /// Phase I/II biotransformation & metabolite toxicity escalation guard.
    #[serde(default)]
    pub metabolite_guard: Option<MetaboliteGuardReport>,
}

/// Side-by-side ADMET delta between two molecules (e.g. parent vs analogue / bioisostere).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdmetComparison {
    pub name_a: String,
    pub name_b: String,
    pub smiles_a: String,
    pub smiles_b: String,
    pub delta_mw: f64,
    pub delta_tpsa: f64,
    pub delta_wlogp: f64,
    pub delta_consensus_logp: f64,
    pub delta_logd_ph74: f64,
    pub delta_esol_log_s: f64,
    pub delta_qed: f64,
    pub delta_cns_mpo: f64,
    pub delta_sa: f64,
    pub alerts_added: Vec<String>,
    pub alerts_removed: Vec<String>,
    pub eval_a: MoleculeEvaluation,
    pub eval_b: MoleculeEvaluation,
}
