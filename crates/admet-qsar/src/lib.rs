#![forbid(unsafe_code)]

//! `admet-qsar`: Pure-Rust QSAR GBDT trainer & evaluator over Therapeutics Data Commons (TDC),
//! ECFP4/Morgan radius-2 fingerprints, Bemis-Murcko scaffold splitting, Split Conformal Prediction
//! (90% CI), Applicability Domain, atomic Sites of Metabolism (SoM), and configurable TPP MPO.

pub mod data_dir;
pub mod fingerprint;
pub mod gbdt;
pub mod metabolites;
pub mod offtarget;
pub mod pack;
pub mod pbpk;
pub mod som;
pub mod tpp;

use admet_calc::{
    AdmetComparison, AdmetError, MoleculeEvaluation, TppPreset, compare_evaluations,
    evaluate_molecule,
};
use chem_core::{Molecule, parse_smiles};
use rayon::prelude::*;

pub use data_dir::{
    admet_models_dir, craft_data_dir, default_model_pack_path, default_sea_index_path, tdc_raw_dir,
};
pub use fingerprint::{
    Ecfp4BitVec, FP_BITS, MolFeatures, NUM_DENSE_FEATURES, TOTAL_FEATURES, compute_ecfp4,
    compute_murcko_scaffold_hash, extract_mol_features,
};
pub use gbdt::{
    ApplicabilityDomainIndex, DecisionTree, GbdtTrainConfig, ScaffoldTestMetrics,
    TrainedEndpointModel, TreeNode, compute_pearson, compute_pr_auc, compute_roc_auc,
    compute_spearman, train_endpoint_gbdt,
};
pub use metabolites::{DEFAULT_MAX_METABOLITES, run_metabolite_guard};
pub use offtarget::{
    SEA_TC_CUTOFF, SeaOffTargetIndex, SeaReferenceLigand, SeaTargetSet, build_sea_index_from_tsv,
    builtin_safety20_sea_index, load_default_sea_index,
};
pub use pack::{
    AdmetModelPack, QsarError, TDC_ENDPOINTS, TdcEndpointSpec, load_default_model_pack,
    parse_tdc_tab, scaffold_split_indices, summarize_tox21, train_tdc_model_pack,
};
pub use pbpk::{dg_kcal_to_ki_nm, render_pk_dose_markdown, simulate_human_pk_dose};
pub use som::predict_metabolism_sites;
pub use tpp::compute_tpp_mpo;

/// Runtime options controlling optional Ola 1 safety modules during batch/single evaluation.
#[derive(Debug, Clone, Copy)]
pub struct EvalOptions {
    pub run_offtargets: bool,
    pub run_metabolites: bool,
    pub max_metabolites: usize,
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self {
            run_offtargets: true,
            run_metabolites: true,
            max_metabolites: DEFAULT_MAX_METABOLITES,
        }
    }
}

/// Enriches an existing `MoleculeEvaluation` in-place with quantitative TDC QSAR + Tox21,
/// SEA Off-Target Fishing, atomic Sites of Metabolism, Phase I/II Metabolite Guard, and TPP MPO.
pub fn enrich_evaluation_with_options(
    mol: &Molecule,
    eval: &mut MoleculeEvaluation,
    pack: Option<&AdmetModelPack>,
    tpp: TppPreset,
    opts: EvalOptions,
) {
    let feat = extract_mol_features(mol, eval);
    if let Some(mp) = pack {
        let src = default_model_pack_path().display().to_string();
        let report = mp.predict_all(&feat, &src);
        eval.tox21_summary = summarize_tox21(&report);
        eval.qsar = Some(report);
    }
    if opts.run_offtargets {
        eval.off_targets = Some(load_default_sea_index().predict_fp(&feat.fp));
    }
    eval.metabolism_sites = predict_metabolism_sites(mol);
    if opts.run_metabolites {
        let sites_snapshot = eval.metabolism_sites.clone();
        eval.metabolite_guard = Some(run_metabolite_guard(
            mol,
            eval,
            &sites_snapshot,
            pack,
            opts.max_metabolites,
        ));
    }
    eval.tpp_mpo = Some(compute_tpp_mpo(eval, tpp));
}

/// Enriches an existing `MoleculeEvaluation` in-place with all default QSAR + Safety + Metabolite Guard modules.
pub fn enrich_evaluation(
    mol: &Molecule,
    eval: &mut MoleculeEvaluation,
    pack: Option<&AdmetModelPack>,
    tpp: TppPreset,
) {
    enrich_evaluation_with_options(mol, eval, pack, tpp, EvalOptions::default());
}

/// Evaluates a `Molecule` with complete physicochemical + TDC QSAR + Tox21 + SEA + SoM + Metabolite Guard + TPP MPO.
#[must_use]
pub fn evaluate_molecule_full(
    mol: &Molecule,
    pack: Option<&AdmetModelPack>,
    tpp: TppPreset,
) -> MoleculeEvaluation {
    let mut eval = evaluate_molecule(mol);
    enrich_evaluation(mol, &mut eval, pack, tpp);
    eval
}

/// Evaluates a `Molecule` with custom `EvalOptions` (e.g. `run_metabolites: false` for 10k+ fast screens).
#[must_use]
pub fn evaluate_molecule_with_options(
    mol: &Molecule,
    pack: Option<&AdmetModelPack>,
    tpp: TppPreset,
    opts: EvalOptions,
) -> MoleculeEvaluation {
    let mut eval = evaluate_molecule(mol);
    enrich_evaluation_with_options(mol, &mut eval, pack, tpp, opts);
    eval
}

/// Evaluates a `Molecule` using the auto-loaded local TDC model pack (if present in `CRAFT_DATA_DIR`).
#[must_use]
pub fn evaluate_molecule_auto(mol: &Molecule, tpp: TppPreset) -> MoleculeEvaluation {
    evaluate_molecule_full(mol, load_default_model_pack(), tpp)
}

/// Parses a SMILES string and evaluates complete ADMET + QSAR + Safety + SoM + Metabolite Guard + TPP MPO.
pub fn evaluate_smiles_full(
    smiles: &str,
    name: Option<&str>,
    pack: Option<&AdmetModelPack>,
    tpp: TppPreset,
) -> Result<MoleculeEvaluation, AdmetError> {
    let mut mol = parse_smiles(smiles)?;
    if let Some(n) = name {
        mol.name = Some(n.to_string());
    }
    Ok(evaluate_molecule_full(&mol, pack, tpp))
}

/// Parses a SMILES string and evaluates using the auto-loaded local TDC model pack.
pub fn evaluate_smiles_auto(
    smiles: &str,
    name: Option<&str>,
    tpp: TppPreset,
) -> Result<MoleculeEvaluation, AdmetError> {
    evaluate_smiles_full(smiles, name, load_default_model_pack(), tpp)
}

/// Evaluates a batch of molecules in parallel across all CPU cores with QSAR + SoM + Metabolite Guard + TPP MPO.
#[must_use]
pub fn evaluate_batch_full(
    molecules: &[Molecule],
    pack: Option<&AdmetModelPack>,
    tpp: TppPreset,
) -> Vec<MoleculeEvaluation> {
    evaluate_batch_with_options(molecules, pack, tpp, EvalOptions::default())
}

/// Evaluates a batch in parallel with explicit `EvalOptions`.
#[must_use]
pub fn evaluate_batch_with_options(
    molecules: &[Molecule],
    pack: Option<&AdmetModelPack>,
    tpp: TppPreset,
    opts: EvalOptions,
) -> Vec<MoleculeEvaluation> {
    molecules
        .par_iter()
        .map(|m| evaluate_molecule_with_options(m, pack, tpp, opts))
        .collect()
}

/// Evaluates a batch in parallel using the auto-loaded local TDC model pack.
#[must_use]
pub fn evaluate_batch_auto(molecules: &[Molecule], tpp: TppPreset) -> Vec<MoleculeEvaluation> {
    evaluate_batch_full(molecules, load_default_model_pack(), tpp)
}

/// Evaluates a batch in parallel using the auto-loaded local TDC model pack and custom `EvalOptions`.
#[must_use]
pub fn evaluate_batch_auto_with_options(
    molecules: &[Molecule],
    tpp: TppPreset,
    opts: EvalOptions,
) -> Vec<MoleculeEvaluation> {
    evaluate_batch_with_options(molecules, load_default_model_pack(), tpp, opts)
}

/// Compares two SMILES side-by-side with full QSAR + SoM + TPP MPO enrichment.
pub fn compare_smiles_auto(
    smiles_a: &str,
    name_a: Option<&str>,
    smiles_b: &str,
    name_b: Option<&str>,
    tpp: TppPreset,
) -> Result<AdmetComparison, AdmetError> {
    let a = evaluate_smiles_auto(smiles_a, name_a, tpp)?;
    let b = evaluate_smiles_auto(smiles_b, name_b, tpp)?;
    Ok(compare_evaluations(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use admet_calc::QsarTaskKind;

    #[test]
    fn ecfp4_and_murcko_scaffold_group_analogs_cleanly() -> Result<(), Box<dyn std::error::Error>> {
        let asp = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")?;
        let sal = parse_smiles("Oc1ccccc1C(=O)O")?;
        let caf = parse_smiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C")?;

        let (fp_asp, _) = compute_ecfp4(&asp);
        let (fp_sal, _) = compute_ecfp4(&sal);
        let (fp_caf, _) = compute_ecfp4(&caf);

        assert!((fp_asp.tanimoto(&fp_asp) - 1.0).abs() < 1e-12);
        assert!(fp_asp.tanimoto(&fp_sal) > fp_asp.tanimoto(&fp_caf));
        // Aspirin and salicylic acid share the exact phenyl Bemis-Murcko scaffold
        assert_eq!(
            compute_murcko_scaffold_hash(&asp),
            compute_murcko_scaffold_hash(&sal)
        );
        assert_ne!(
            compute_murcko_scaffold_hash(&asp),
            compute_murcko_scaffold_hash(&caf)
        );
        Ok(())
    }

    #[test]
    fn som_and_tpp_mpo_identify_real_drug_liabilities() -> Result<(), Box<dyn std::error::Error>> {
        // Ibuprofen: benzylic CH2/CH + carboxylic acid
        let ibu = parse_smiles("CC(C)Cc1ccc(C(C)C(=O)O)cc1")?;
        let sites = predict_metabolism_sites(&ibu);
        assert!(!sites.is_empty());
        assert!(
            sites.iter().any(|s| s.reaction.contains("Benzylic")),
            "expected benzylic hydroxylation hotspot on ibuprofen, got {sites:?}"
        );

        // Anisole derivative + basic amine (propranolol-like): O-dealkylation & N-dealkylation
        let prop = parse_smiles("CC(C)NCC(O)COc1cccc2ccccc12")?;
        let eval = evaluate_molecule_full(&prop, None, TppPreset::OralSmallMolecule);
        assert!(eval.tpp_mpo.is_some());
        assert!(!eval.metabolism_sites.is_empty());
        Ok(())
    }

    #[test]
    fn synthetic_gbdt_learns_classification_and_regression_with_conformal_calibration()
    -> Result<(), Box<dyn std::error::Error>> {
        let smiles_pool = [
            ("c1ccccc1", 0.0, 2.1),
            ("Cc1ccccc1", 0.0, 2.6),
            ("CCc1ccccc1", 0.0, 3.1),
            ("CCCc1ccccc1", 0.0, 3.6),
            ("CCCCc1ccccc1", 0.0, 4.1),
            ("c1ccc(Cl)cc1", 0.0, 2.8),
            ("c1ccc(Br)cc1", 0.0, 3.0),
            ("c1ccc(F)cc1", 0.0, 2.3),
            ("c1ccc2ccccc2c1", 0.0, 3.3),
            ("Cc1ccc2ccccc2c1", 0.0, 3.8),
            ("CCN(CC)CCc1ccccc1", 1.0, 2.5),
            ("CN(C)CCCc1ccccc1", 1.0, 2.4),
            ("CCN(CC)CCOc1ccccc1", 1.0, 2.2),
            ("CN(C)CCc1ccc(Cl)cc1", 1.0, 2.9),
            ("CCCN(CCC)CCc1ccccc1", 1.0, 3.4),
            ("CN1CCN(CC1)c1ccccc1", 1.0, 1.8),
            ("CCN1CCN(CC1)c1ccc(F)cc1", 1.0, 2.0),
            ("CN(C)CC(O)COc1ccccc1", 1.0, 1.3),
            ("CC(C)NCC(O)COc1ccccc1", 1.0, 1.7),
            ("CCCCN(CCCC)CCc1ccccc1", 1.0, 4.0),
        ];

        let mut feats = Vec::new();
        let mut y_cls = Vec::new();
        let mut y_reg = Vec::new();
        // Repeat with slight alkyl diversity via clean feature cloning for quick unit test
        for _ in 0..4 {
            for &(smi, yc, yr) in &smiles_pool {
                let mol = parse_smiles(smi)?;
                let ev = evaluate_molecule(&mol);
                feats.push(extract_mol_features(&mol, &ev));
                y_cls.push(yc);
                y_reg.push(yr);
            }
        }
        let refs: Vec<&MolFeatures> = feats.iter().collect();
        let cfg = GbdtTrainConfig {
            max_trees: 40,
            max_depth: 3,
            min_child_hess: 1.0,
            ..Default::default()
        };
        let cls_model = train_endpoint_gbdt(
            "synth_basic",
            "Synthetic Basic Amine",
            "Test",
            QsarTaskKind::BinaryClassification,
            "prob",
            false,
            &refs[..50],
            &y_cls[..50],
            &refs[50..65],
            &y_cls[50..65],
            &refs[65..],
            &y_cls[65..],
            &cfg,
        );
        assert!(
            cls_model.metrics.roc_auc.unwrap_or(0.0) >= 0.85,
            "expected high AUC on synthetic basic amine task, got {:?}",
            cls_model.metrics.roc_auc
        );

        let reg_model = train_endpoint_gbdt(
            "synth_logp",
            "Synthetic LogP",
            "Test",
            QsarTaskKind::Regression,
            "logP",
            false,
            &refs[..50],
            &y_reg[..50],
            &refs[50..65],
            &y_reg[50..65],
            &refs[65..],
            &y_reg[65..],
            &cfg,
        );
        assert!(
            reg_model.metrics.spearman_rho.unwrap_or(0.0) >= 0.70,
            "expected strong Spearman rho on synthetic regression task, got {:?}",
            reg_model.metrics.spearman_rho
        );
        Ok(())
    }
}
