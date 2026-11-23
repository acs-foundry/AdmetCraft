#![forbid(unsafe_code)]

use admet_calc::{
    ApplicabilityDomain, ApplicabilityDomainStatus, QsarEndpointPrediction, QsarTaskKind,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::fingerprint::{Ecfp4BitVec, MolFeatures, NUM_DENSE_FEATURES, TOTAL_FEATURES};

/// Compact decision-tree node (`feature == u16::MAX` marks a leaf holding `value`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TreeNode {
    /// Feature index in `0..TOTAL_FEATURES`, or `u16::MAX` for a leaf node.
    pub feature: u16,
    /// Split threshold (`x[feature] <= threshold` goes left) or leaf weight when leaf.
    pub value: f32,
    /// Left child index (unused for leaf).
    pub left: u16,
    /// Right child index (unused for leaf).
    pub right: u16,
}

/// Single regression/logit decision tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    pub nodes: Vec<TreeNode>,
}

impl DecisionTree {
    #[inline]
    #[must_use]
    pub fn predict(&self, feat: &MolFeatures) -> f64 {
        let mut idx = 0usize;
        for _ in 0..32 {
            let Some(node) = self.nodes.get(idx) else {
                return 0.0;
            };
            if node.feature == u16::MAX {
                return f64::from(node.value);
            }
            let x = feat.get(usize::from(node.feature));
            idx = if x <= node.value {
                usize::from(node.left)
            } else {
                usize::from(node.right)
            };
        }
        0.0
    }
}

/// Honest held-out Bemis-Murcko scaffold-split test metrics recorded at training time.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ScaffoldTestMetrics {
    pub train_n: usize,
    pub cal_n: usize,
    pub test_n: usize,
    pub num_trees_used: usize,
    /// Classification metrics (populated for `BinaryClassification`).
    pub roc_auc: Option<f64>,
    pub pr_auc: Option<f64>,
    pub brier_score: Option<f64>,
    /// Regression metrics (populated for `Regression`).
    pub spearman_rho: Option<f64>,
    pub pearson_r: Option<f64>,
    pub rmse: Option<f64>,
    pub mae: Option<f64>,
    /// Empirical coverage of the 90% split-conformal interval/set on the held-out scaffold test fold.
    pub conformal_90_coverage: f64,
    /// Fraction of classification test predictions yielding a decisive singleton set (`{Pos}` or `{Neg}`).
    pub conformal_singleton_rate: Option<f64>,
}

/// Compact Applicability Domain index stored alongside each trained endpoint.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ApplicabilityDomainIndex {
    /// Representative training fingerprints (compact `[u64; 16]` ECFP4 words).
    pub ref_fps: Vec<Ecfp4BitVec>,
    /// Mean of first 20 core physicochemical descriptors on training set.
    pub desc_mean: Vec<f32>,
    /// Std-dev of first 20 core physicochemical descriptors on training set.
    pub desc_std: Vec<f32>,
}

impl ApplicabilityDomainIndex {
    #[must_use]
    pub fn build(train_feats: &[&MolFeatures], max_refs: usize) -> Self {
        let n = train_feats.len();
        let mut ref_fps = Vec::with_capacity(n.min(max_refs));
        if n <= max_refs {
            for f in train_feats {
                ref_fps.push(f.fp);
            }
        } else {
            // Deterministic stratified stride across training set
            let step = (n as f64) / (max_refs as f64);
            for k in 0..max_refs {
                let idx = ((k as f64) * step) as usize;
                if let Some(f) = train_feats.get(idx.min(n - 1)) {
                    ref_fps.push(f.fp);
                }
            }
        }

        let d_dim = 20.min(NUM_DENSE_FEATURES);
        let mut desc_mean = vec![0.0f32; d_dim];
        let mut desc_std = vec![1.0f32; d_dim];
        if n > 1 {
            let nf = n as f64;
            for j in 0..d_dim {
                let sum: f64 = train_feats.iter().map(|f| f64::from(f.dense[j])).sum();
                let mu = sum / nf;
                let var: f64 = train_feats
                    .iter()
                    .map(|f| {
                        let diff = f64::from(f.dense[j]) - mu;
                        diff * diff
                    })
                    .sum::<f64>()
                    / nf;
                desc_mean[j] = mu as f32;
                desc_std[j] = var.sqrt().max(1e-3) as f32;
            }
        }

        Self {
            ref_fps,
            desc_mean,
            desc_std,
        }
    }

    #[must_use]
    pub fn evaluate(&self, feat: &MolFeatures) -> ApplicabilityDomain {
        let mut top5 = [0.0f64; 5];
        for rfp in &self.ref_fps {
            let sim = feat.fp.tanimoto(rfp);
            if sim > top5[4] {
                top5[4] = sim;
                let mut k = 4;
                while k > 0 && top5[k] > top5[k - 1] {
                    top5.swap(k, k - 1);
                    k -= 1;
                }
            }
        }
        let max_tanimoto = top5[0];
        let k_eff = self.ref_fps.len().clamp(1, 5);
        let top5_tanimoto = top5[..k_eff].iter().sum::<f64>() / (k_eff as f64);

        let mut z2_sum = 0.0f64;
        let d_dim = self.desc_mean.len().min(NUM_DENSE_FEATURES);
        for j in 0..d_dim {
            let z = (f64::from(feat.dense[j]) - f64::from(self.desc_mean[j]))
                / f64::from(self.desc_std[j]).max(1e-3);
            z2_sum += z.clamp(-10.0, 10.0).powi(2);
        }
        let descriptor_z_rms = if d_dim > 0 {
            (z2_sum / (d_dim as f64)).sqrt()
        } else {
            0.0
        };

        let status = if (max_tanimoto >= 0.40 || top5_tanimoto >= 0.30) && descriptor_z_rms <= 3.2 {
            ApplicabilityDomainStatus::InDomain
        } else if max_tanimoto >= 0.25 && descriptor_z_rms <= 4.5 {
            ApplicabilityDomainStatus::Borderline
        } else {
            ApplicabilityDomainStatus::OutOfDomain
        };

        ApplicabilityDomain {
            status,
            max_tanimoto: round3(max_tanimoto),
            top5_tanimoto: round3(top5_tanimoto),
            descriptor_z_rms: round2(descriptor_z_rms),
        }
    }
}

/// Complete trained pure-Rust GBDT model for one TDC ADMET endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainedEndpointModel {
    pub endpoint_id: String,
    pub display_name: String,
    pub category: String,
    pub task: QsarTaskKind,
    pub units: String,
    /// True when regression targets were modeled in `log10(y)` space (VDss, CLint, Half-Life).
    pub log10_transform: bool,
    pub base_score: f64,
    pub learning_rate: f64,
    /// Platt temperature scaling parameters for classification (`sigmoid(platt_a * raw + platt_b)`).
    pub platt_a: f64,
    pub platt_b: f64,
    /// Split-conformal 90% residual quantile (regression).
    pub conformal_q90_reg: f64,
    /// Mondrian split-conformal 90% nonconformity quantiles `(q_neg, q_pos)` (classification).
    pub conformal_q90_neg: f64,
    pub conformal_q90_pos: f64,
    pub trees: Vec<DecisionTree>,
    pub ad_index: ApplicabilityDomainIndex,
    pub metrics: ScaffoldTestMetrics,
}

impl TrainedEndpointModel {
    #[inline]
    #[must_use]
    pub fn raw_margin(&self, feat: &MolFeatures) -> f64 {
        let mut s = self.base_score;
        for t in &self.trees {
            s += self.learning_rate * t.predict(feat);
        }
        s
    }

    /// Predict value/probability + 90% Split Conformal interval/set + Applicability Domain.
    #[must_use]
    pub fn predict(&self, feat: &MolFeatures) -> QsarEndpointPrediction {
        let raw = self.raw_margin(feat);
        let ad = self.ad_index.evaluate(feat);
        // Honest uncertainty inflation when moving out of the applicability domain
        let ad_inflate = match ad.status {
            ApplicabilityDomainStatus::InDomain => 1.0,
            ApplicabilityDomainStatus::Borderline => 1.25,
            ApplicabilityDomainStatus::OutOfDomain => 1.55,
        };

        match self.task {
            QsarTaskKind::BinaryClassification => {
                let prob = sigmoid(self.platt_a * raw + self.platt_b).clamp(0.001, 0.999);
                // Mondrian 90% conformal prediction set
                let in_pos = (1.0 - prob) <= (self.conformal_q90_pos * ad_inflate).min(0.999);
                let in_neg = prob <= (self.conformal_q90_neg * ad_inflate).min(0.999);
                let conformal_call = match (in_pos, in_neg) {
                    (true, false) => "Positive (90% Conf.)".to_string(),
                    (false, true) => "Negative (90% Conf.)".to_string(),
                    (true, true) => "Uncertain (Both)".to_string(),
                    (false, false) => {
                        if prob >= 0.5 {
                            "Positive (OOD)".to_string()
                        } else {
                            "Negative (OOD)".to_string()
                        }
                    }
                };
                // Informative probability uncertainty band around calibrated probability
                let half_band =
                    (0.5 * (self.conformal_q90_pos + self.conformal_q90_neg) * 0.35 * ad_inflate)
                        .clamp(0.04, 0.42);
                let lo = (prob - half_band).clamp(0.0, 1.0);
                let hi = (prob + half_band).clamp(0.0, 1.0);

                QsarEndpointPrediction {
                    endpoint_id: self.endpoint_id.clone(),
                    display_name: self.display_name.clone(),
                    category: self.category.clone(),
                    task: self.task,
                    value: round3(prob),
                    ci90_low: round3(lo),
                    ci90_high: round3(hi),
                    units: self.units.clone(),
                    linear_value: None,
                    linear_ci90: None,
                    conformal_call,
                    applicability_domain: ad,
                    test_metric_name: "ROC AUC".to_string(),
                    test_metric_value: round3(self.metrics.roc_auc.unwrap_or(0.5)),
                    test_conformal_coverage: round3(self.metrics.conformal_90_coverage),
                }
            }
            QsarTaskKind::Regression => {
                let q = self.conformal_q90_reg * ad_inflate;
                let (pt, lo, hi) = if self.units == "%" {
                    (
                        raw.clamp(0.0, 100.0),
                        (raw - q).clamp(0.0, 100.0),
                        (raw + q).clamp(0.0, 100.0),
                    )
                } else {
                    (raw, raw - q, raw + q)
                };
                let (linear_value, linear_ci90, call_str) = if self.log10_transform {
                    let lin = 10.0_f64.powf(pt.clamp(-4.0, 5.0));
                    let lin_lo = 10.0_f64.powf(lo.clamp(-4.0, 5.0));
                    let lin_hi = 10.0_f64.powf(hi.clamp(-4.0, 5.0));
                    (
                        Some(round2(lin)),
                        Some([round2(lin_lo), round2(lin_hi)]),
                        format!("{:.2} [{:.2}..{:.2}] {}", lin, lin_lo, lin_hi, self.units),
                    )
                } else {
                    (
                        None,
                        None,
                        format!("{:.2} [{:.2}..{:.2}] {}", pt, lo, hi, self.units),
                    )
                };
                QsarEndpointPrediction {
                    endpoint_id: self.endpoint_id.clone(),
                    display_name: self.display_name.clone(),
                    category: self.category.clone(),
                    task: self.task,
                    value: round3(pt),
                    ci90_low: round3(lo),
                    ci90_high: round3(hi),
                    units: self.units.clone(),
                    linear_value,
                    linear_ci90,
                    conformal_call: call_str,
                    applicability_domain: ad,
                    test_metric_name: "Spearman rho".to_string(),
                    test_metric_value: round3(self.metrics.spearman_rho.unwrap_or(0.0)),
                    test_conformal_coverage: round3(self.metrics.conformal_90_coverage),
                }
            }
        }
    }
}

/// Hyperparameters for training a pure-Rust histogram GBDT model.
#[derive(Debug, Clone)]
pub struct GbdtTrainConfig {
    pub max_trees: usize,
    pub max_depth: usize,
    pub learning_rate: f64,
    pub l2_reg: f64,
    pub min_child_hess: f64,
    pub min_split_gain: f64,
    pub col_subsample: f64,
    pub row_subsample: f64,
    pub early_stopping_rounds: usize,
    pub max_ad_refs: usize,
    pub seed: u64,
}

impl Default for GbdtTrainConfig {
    fn default() -> Self {
        Self {
            max_trees: 240,
            max_depth: 5,
            learning_rate: 0.05,
            l2_reg: 3.0,
            min_child_hess: 3.0,
            min_split_gain: 0.02,
            col_subsample: 0.30,
            row_subsample: 0.80,
            early_stopping_rounds: 25,
            max_ad_refs: 1_200,
            seed: 2026_1119,
        }
    }
}

/// Simple deterministic XorShift64* PRNG (zero external RNG crate required).
struct FastRng {
    state: u64,
}

impl FastRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.max(0x9e37_79b9_7f4a_7c15),
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

#[inline]
fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

const MAX_CONT_BINS: usize = 32;

/// Pre-binned column-major training matrix for blazing-fast Rayon histogram GBDT splitting.
struct BinnedMatrix {
    /// Column-major bin index (`col * n_rows + row`) in `0..n_bins[col]`.
    bins: Vec<u8>,
    /// Upper threshold value for each bin of each feature (`thresholds[col][b]`).
    thresholds: Vec<Vec<f32>>,
    n_rows: usize,
}

impl BinnedMatrix {
    fn build(train_feats: &[&MolFeatures]) -> Self {
        let n_rows = train_feats.len();
        let mut thresholds: Vec<Vec<f32>> = vec![Vec::new(); TOTAL_FEATURES];
        let mut bins = vec![0u8; TOTAL_FEATURES * n_rows];

        // Parallel per-feature quantile binning
        let cols: Vec<(Vec<f32>, Vec<u8>)> = (0..TOTAL_FEATURES)
            .into_par_iter()
            .map(|col| {
                if col < NUM_DENSE_FEATURES {
                    let mut vals: Vec<f32> = train_feats.iter().map(|f| f.dense[col]).collect();
                    vals.sort_by(f32::total_cmp);
                    let mut edges: Vec<f32> = Vec::with_capacity(MAX_CONT_BINS);
                    if !vals.is_empty() {
                        for b in 1..MAX_CONT_BINS {
                            let qi = (b * vals.len()) / MAX_CONT_BINS;
                            let v = vals[qi.min(vals.len() - 1)];
                            if edges.last().is_none_or(|&prev| v > prev + 1e-6) {
                                edges.push(v);
                            }
                        }
                        if let Some(&last_v) = vals.last()
                            && edges.last().is_none_or(|&prev| last_v > prev + 1e-6)
                        {
                            edges.push(last_v);
                        }
                    }
                    if edges.is_empty() {
                        edges.push(0.0);
                    }
                    let mut col_bins = vec![0u8; n_rows];
                    for (r, f) in train_feats.iter().enumerate() {
                        let x = f.dense[col];
                        let b = edges.partition_point(|&e| e < x).min(edges.len() - 1);
                        col_bins[r] = b as u8;
                    }
                    (edges, col_bins)
                } else {
                    // ECFP4 count features take small integer values 0..=8 -> exact bins!
                    let fp_idx = col - NUM_DENSE_FEATURES;
                    let mut max_c = 0u8;
                    for f in train_feats {
                        if f.fp_counts[fp_idx] > max_c {
                            max_c = f.fp_counts[fp_idx];
                        }
                    }
                    let max_b = usize::from(max_c.min(4));
                    let mut edges = Vec::with_capacity(max_b + 1);
                    for k in 0..=max_b {
                        edges.push((k as f32) + 0.5);
                    }
                    let mut col_bins = vec![0u8; n_rows];
                    for (r, f) in train_feats.iter().enumerate() {
                        col_bins[r] = f.fp_counts[fp_idx].min(max_b as u8);
                    }
                    (edges, col_bins)
                }
            })
            .collect();

        for (col, (edges, col_bins)) in cols.into_iter().enumerate() {
            thresholds[col] = edges;
            let dst = &mut bins[col * n_rows..(col + 1) * n_rows];
            dst.copy_from_slice(&col_bins);
        }

        Self {
            bins,
            thresholds,
            n_rows,
        }
    }
}

/// Trains one endpoint model using strict `train / cal / test` splits (caller guarantees
/// Bemis-Murcko scaffold separation), performs Platt + Split Conformal calibration on `cal`,
/// and evaluates honest held-out performance on `test`.
#[allow(clippy::too_many_arguments)]
pub fn train_endpoint_gbdt(
    endpoint_id: &str,
    display_name: &str,
    category: &str,
    task: QsarTaskKind,
    units: &str,
    log10_transform: bool,
    train_feats: &[&MolFeatures],
    train_y: &[f64],
    cal_feats: &[&MolFeatures],
    cal_y: &[f64],
    test_feats: &[&MolFeatures],
    test_y: &[f64],
    cfg: &GbdtTrainConfig,
) -> TrainedEndpointModel {
    let n_train = train_feats.len();
    let n_cal = cal_feats.len();
    let n_test = test_feats.len();

    // Base score (log-odds prior for binary, mean for regression)
    let base_score = if n_train == 0 {
        0.0
    } else {
        match task {
            QsarTaskKind::BinaryClassification => {
                let p = (train_y.iter().sum::<f64>() / (n_train as f64)).clamp(0.01, 0.99);
                (p / (1.0 - p)).ln()
            }
            QsarTaskKind::Regression => train_y.iter().sum::<f64>() / (n_train as f64),
        }
    };

    // Moderate inverse-frequency class weight for imbalanced classification (capped at 3.0)
    let pos_weight = if task == QsarTaskKind::BinaryClassification && n_train > 0 {
        let n_pos = train_y.iter().filter(|&&y| y > 0.5).count() as f64;
        let n_neg = (n_train as f64) - n_pos;
        if n_pos > 1.0 && n_neg > 1.0 {
            (n_neg / n_pos).sqrt().clamp(0.5, 2.5)
        } else {
            1.0
        }
    } else {
        1.0
    };

    let bmat = BinnedMatrix::build(train_feats);
    let mut train_pred = vec![base_score; n_train];
    let mut cal_pred = vec![base_score; n_cal];
    let mut grad = vec![0.0f64; n_train];
    let mut hess = vec![0.0f64; n_train];

    let mut rng = FastRng::new(cfg.seed);
    let mut trees: Vec<DecisionTree> = Vec::with_capacity(cfg.max_trees);
    let mut best_cal_loss = f64::INFINITY;
    let mut best_num_trees = 0usize;
    let mut stale_rounds = 0usize;

    // Active non-constant features
    let active_cols: Vec<usize> = (0..TOTAL_FEATURES)
        .filter(|&c| bmat.thresholds[c].len() >= 2)
        .collect();

    for _round in 0..cfg.max_trees {
        // Compute first & second derivatives on training set
        match task {
            QsarTaskKind::BinaryClassification => {
                for i in 0..n_train {
                    let p = sigmoid(train_pred[i]);
                    let w = if train_y[i] > 0.5 { pos_weight } else { 1.0 };
                    grad[i] = w * (p - train_y[i]);
                    hess[i] = (w * p * (1.0 - p)).max(1e-5);
                }
            }
            QsarTaskKind::Regression => {
                for i in 0..n_train {
                    grad[i] = train_pred[i] - train_y[i];
                    hess[i] = 1.0;
                }
            }
        }

        // Row subsampling
        let mut root_rows = Vec::with_capacity(n_train);
        for i in 0..n_train {
            if rng.next_f64() <= cfg.row_subsample {
                root_rows.push(i);
            }
        }
        if root_rows.len() < 8 {
            root_rows = (0..n_train).collect();
        }

        // Column subsampling: always keep ~70% of dense descriptors + subsampled ECFP4 bits
        let mut tree_cols = Vec::with_capacity(active_cols.len() / 3 + 48);
        for &c in &active_cols {
            let keep_p = if c < NUM_DENSE_FEATURES {
                0.75
            } else {
                cfg.col_subsample
            };
            if rng.next_f64() <= keep_p {
                tree_cols.push(c);
            }
        }
        if tree_cols.is_empty() {
            tree_cols = active_cols.clone();
        }

        let tree = build_tree(&bmat, &grad, &hess, &root_rows, &tree_cols, cfg);

        // Update train & calibration predictions
        for i in 0..n_train {
            train_pred[i] += cfg.learning_rate * tree.predict(train_feats[i]);
        }
        for i in 0..n_cal {
            cal_pred[i] += cfg.learning_rate * tree.predict(cal_feats[i]);
        }
        trees.push(tree);

        // Early stopping check on scaffold-held-out calibration set
        let cal_loss = if n_cal == 0 {
            0.0
        } else {
            match task {
                QsarTaskKind::BinaryClassification => {
                    let mut sum = 0.0;
                    for i in 0..n_cal {
                        let p = sigmoid(cal_pred[i]).clamp(1e-6, 1.0 - 1e-6);
                        sum -= cal_y[i] * p.ln() + (1.0 - cal_y[i]) * (1.0 - p).ln();
                    }
                    sum / (n_cal as f64)
                }
                QsarTaskKind::Regression => {
                    let mut sum = 0.0;
                    for i in 0..n_cal {
                        let e = cal_pred[i] - cal_y[i];
                        sum += e * e;
                    }
                    (sum / (n_cal as f64)).sqrt()
                }
            }
        };

        if cal_loss + 1e-5 < best_cal_loss {
            best_cal_loss = cal_loss;
            best_num_trees = trees.len();
            stale_rounds = 0;
        } else {
            stale_rounds += 1;
            if stale_rounds >= cfg.early_stopping_rounds && trees.len() >= 30 {
                break;
            }
        }
    }

    if best_num_trees > 0 && best_num_trees < trees.len() {
        trees.truncate(best_num_trees);
    }

    // Recompute exact calibration margins with the truncated ensemble
    let cal_raw: Vec<f64> = cal_feats
        .iter()
        .map(|f| {
            let mut s = base_score;
            for t in &trees {
                s += cfg.learning_rate * t.predict(f);
            }
            s
        })
        .collect();

    // Platt scaling (for classification) + Split Conformal 90% quantiles on calibration fold
    let (platt_a, platt_b, conformal_q90_reg, conformal_q90_neg, conformal_q90_pos) = match task {
        QsarTaskKind::BinaryClassification => {
            let (pa, pb) = fit_platt_scaling(&cal_raw, cal_y);
            let mut neg_scores = Vec::new();
            let mut pos_scores = Vec::new();
            for (&raw, &y) in cal_raw.iter().zip(cal_y.iter()) {
                let p = sigmoid(pa * raw + pb).clamp(0.001, 0.999);
                if y > 0.5 {
                    pos_scores.push(1.0 - p);
                } else {
                    neg_scores.push(p);
                }
            }
            // The floor at 0.15 is not a detail of numerical hygiene: it decides what the
            // reported coverage means. Split conformal guarantees ~90% coverage *at the
            // quantile it computes*; raising that quantile to 0.15 widens the prediction set,
            // which pushes measured coverage above 90% while turning more predictions into
            // the uninformative {0,1} set that contains both labels. On the shipped v1 pack,
            // 9 of 25 binary endpoints sit exactly on this floor, mean coverage reads 93.4%,
            // and the mean singleton (informative) rate is 0.538 — so roughly half of the
            // predictions say nothing. Read `conformal_singleton_rate` alongside coverage;
            // coverage on its own is not a quality measure once this floor is in play.
            let q_neg = conformal_quantile(&mut neg_scores, 0.90).clamp(0.15, 0.99);
            let q_pos = conformal_quantile(&mut pos_scores, 0.90).clamp(0.15, 0.99);
            (pa, pb, 0.0, q_neg, q_pos)
        }
        QsarTaskKind::Regression => {
            let mut abs_res: Vec<f64> = cal_raw
                .iter()
                .zip(cal_y.iter())
                .map(|(&pred, &y)| (pred - y).abs())
                .collect();
            let q90 = conformal_quantile(&mut abs_res, 0.90).max(0.05);
            (1.0, 0.0, q90, 0.90, 0.90)
        }
    };

    let ad_index = ApplicabilityDomainIndex::build(train_feats, cfg.max_ad_refs);

    let mut model = TrainedEndpointModel {
        endpoint_id: endpoint_id.to_string(),
        display_name: display_name.to_string(),
        category: category.to_string(),
        task,
        units: units.to_string(),
        log10_transform,
        base_score,
        learning_rate: cfg.learning_rate,
        platt_a,
        platt_b,
        conformal_q90_reg,
        conformal_q90_neg,
        conformal_q90_pos,
        trees,
        ad_index,
        metrics: ScaffoldTestMetrics {
            train_n: n_train,
            cal_n: n_cal,
            test_n: n_test,
            num_trees_used: best_num_trees.max(1),
            ..Default::default()
        },
    };

    // Honest held-out Bemis-Murcko scaffold test evaluation
    if n_test > 0 {
        match task {
            QsarTaskKind::BinaryClassification => {
                let mut probs = Vec::with_capacity(n_test);
                let mut covered = 0usize;
                let mut singletons = 0usize;
                let mut brier = 0.0;
                for (feat, &y) in test_feats.iter().zip(test_y.iter()) {
                    let raw = model.raw_margin(feat);
                    let p = sigmoid(model.platt_a * raw + model.platt_b).clamp(0.001, 0.999);
                    probs.push(p);
                    brier += (p - y).powi(2);
                    let in_pos = (1.0 - p) <= model.conformal_q90_pos;
                    let in_neg = p <= model.conformal_q90_neg;
                    if (y > 0.5 && in_pos) || (y <= 0.5 && in_neg) {
                        covered += 1;
                    }
                    if in_pos ^ in_neg {
                        singletons += 1;
                    }
                }
                model.metrics.roc_auc = Some(round3(compute_roc_auc(&probs, test_y)));
                model.metrics.pr_auc = Some(round3(compute_pr_auc(&probs, test_y)));
                model.metrics.brier_score = Some(round3(brier / (n_test as f64)));
                model.metrics.conformal_90_coverage = round3((covered as f64) / (n_test as f64));
                model.metrics.conformal_singleton_rate =
                    Some(round3((singletons as f64) / (n_test as f64)));
            }
            QsarTaskKind::Regression => {
                let mut preds = Vec::with_capacity(n_test);
                let mut covered = 0usize;
                let mut se = 0.0;
                let mut ae = 0.0;
                for (feat, &y) in test_feats.iter().zip(test_y.iter()) {
                    let pred = model.raw_margin(feat);
                    preds.push(pred);
                    let err = (pred - y).abs();
                    se += err * err;
                    ae += err;
                    if err <= model.conformal_q90_reg {
                        covered += 1;
                    }
                }
                model.metrics.spearman_rho = Some(round3(compute_spearman(&preds, test_y)));
                model.metrics.pearson_r = Some(round3(compute_pearson(&preds, test_y)));
                model.metrics.rmse = Some(round3((se / (n_test as f64)).sqrt()));
                model.metrics.mae = Some(round3(ae / (n_test as f64)));
                model.metrics.conformal_90_coverage = round3((covered as f64) / (n_test as f64));
            }
        }
    }

    model
}

fn build_tree(
    bmat: &BinnedMatrix,
    grad: &[f64],
    hess: &[f64],
    root_rows: &[usize],
    tree_cols: &[usize],
    cfg: &GbdtTrainConfig,
) -> DecisionTree {
    let mut nodes: Vec<TreeNode> = Vec::with_capacity(1 << (cfg.max_depth + 1));
    build_node_recursive(bmat, grad, hess, root_rows, tree_cols, 0, cfg, &mut nodes);
    DecisionTree { nodes }
}

#[allow(clippy::too_many_arguments)]
fn build_node_recursive(
    bmat: &BinnedMatrix,
    grad: &[f64],
    hess: &[f64],
    rows: &[usize],
    tree_cols: &[usize],
    depth: usize,
    cfg: &GbdtTrainConfig,
    nodes: &mut Vec<TreeNode>,
) -> u16 {
    let node_idx = nodes.len() as u16;
    let mut g_tot = 0.0f64;
    let mut h_tot = 0.0f64;
    for &r in rows {
        g_tot += grad[r];
        h_tot += hess[r];
    }
    let leaf_val = (-g_tot / (h_tot + cfg.l2_reg)).clamp(-4.0, 4.0) as f32;
    nodes.push(TreeNode {
        feature: u16::MAX,
        value: leaf_val,
        left: 0,
        right: 0,
    });

    if depth >= cfg.max_depth || rows.len() < 12 || h_tot < 2.0 * cfg.min_child_hess {
        return node_idx;
    }

    let parent_score = (g_tot * g_tot) / (h_tot + cfg.l2_reg);
    let n_rows_total = bmat.n_rows;

    // Parallel histogram split search across subsampled features
    let best_split = tree_cols
        .par_iter()
        .filter_map(|&col| {
            let n_bins = bmat.thresholds[col].len();
            if n_bins < 2 {
                return None;
            }
            let mut hg = [0.0f64; MAX_CONT_BINS];
            let mut hh = [0.0f64; MAX_CONT_BINS];
            let col_slice = &bmat.bins[col * n_rows_total..(col + 1) * n_rows_total];
            for &r in rows {
                let b = usize::from(col_slice[r]);
                if b < n_bins {
                    hg[b] += grad[r];
                    hh[b] += hess[r];
                }
            }
            let mut g_l = 0.0f64;
            let mut h_l = 0.0f64;
            let mut best_gain = cfg.min_split_gain;
            let mut best_b = usize::MAX;
            for b in 0..(n_bins - 1) {
                g_l += hg[b];
                h_l += hh[b];
                let h_r = h_tot - h_l;
                if h_l < cfg.min_child_hess || h_r < cfg.min_child_hess {
                    continue;
                }
                let g_r = g_tot - g_l;
                let gain = 0.5
                    * ((g_l * g_l) / (h_l + cfg.l2_reg) + (g_r * g_r) / (h_r + cfg.l2_reg)
                        - parent_score);
                if gain > best_gain {
                    best_gain = gain;
                    best_b = b;
                }
            }
            if best_b < n_bins - 1 {
                Some((best_gain, col, best_b))
            } else {
                None
            }
        })
        .reduce(
            || (f64::NEG_INFINITY, usize::MAX, usize::MAX),
            |a, b| if b.0 > a.0 { b } else { a },
        );

    if best_split.1 == usize::MAX || best_split.0 <= cfg.min_split_gain {
        return node_idx;
    }

    let (_, split_col, split_bin) = best_split;
    let thresh = bmat.thresholds[split_col][split_bin];
    let col_slice = &bmat.bins[split_col * n_rows_total..(split_col + 1) * n_rows_total];
    let mut left_rows = Vec::with_capacity(rows.len() / 2);
    let mut right_rows = Vec::with_capacity(rows.len() / 2);
    for &r in rows {
        if usize::from(col_slice[r]) <= split_bin {
            left_rows.push(r);
        } else {
            right_rows.push(r);
        }
    }
    if left_rows.is_empty() || right_rows.is_empty() {
        return node_idx;
    }

    let left_id = build_node_recursive(
        bmat,
        grad,
        hess,
        &left_rows,
        tree_cols,
        depth + 1,
        cfg,
        nodes,
    );
    let right_id = build_node_recursive(
        bmat,
        grad,
        hess,
        &right_rows,
        tree_cols,
        depth + 1,
        cfg,
        nodes,
    );

    if let Some(slot) = nodes.get_mut(usize::from(node_idx)) {
        slot.feature = split_col as u16;
        slot.value = thresh;
        slot.left = left_id;
        slot.right = right_id;
    }
    node_idx
}

/// Fits 2-parameter Platt scaling `sigmoid(a * margin + b)` via damped Newton iterations on `cal`.
fn fit_platt_scaling(margins: &[f64], labels: &[f64]) -> (f64, f64) {
    if margins.len() < 20 {
        return (1.0, 0.0);
    }
    let n_pos = labels.iter().filter(|&&y| y > 0.5).count() as f64;
    let n_neg = (labels.len() as f64) - n_pos;
    if n_pos < 3.0 || n_neg < 3.0 {
        return (1.0, 0.0);
    }
    // Platt target smoothing
    let t_pos = (n_pos + 1.0) / (n_pos + 2.0);
    let t_neg = 1.0 / (n_neg + 2.0);

    let mut a = 1.0f64;
    let mut b = 0.0f64;
    let reg = 1e-2;
    for _ in 0..25 {
        let mut g_a = reg * (a - 1.0);
        let mut g_b = reg * b;
        let mut h_aa = reg;
        let mut h_ab = 0.0;
        let mut h_bb = reg;
        for (&m, &y) in margins.iter().zip(labels.iter()) {
            let t = if y > 0.5 { t_pos } else { t_neg };
            let p = sigmoid(a * m + b);
            let d = p - t;
            let w = (p * (1.0 - p)).max(1e-6);
            g_a += d * m;
            g_b += d;
            h_aa += w * m * m;
            h_ab += w * m;
            h_bb += w;
        }
        let det = h_aa * h_bb - h_ab * h_ab;
        if det.abs() < 1e-12 {
            break;
        }
        let da = (h_bb * g_a - h_ab * g_b) / det;
        let db = (-h_ab * g_a + h_aa * g_b) / det;
        a = (a - 0.6 * da).clamp(0.25, 3.0);
        b = (b - 0.6 * db).clamp(-3.0, 3.0);
        if da.abs() + db.abs() < 1e-5 {
            break;
        }
    }
    (a, b)
}

/// Finite-sample split-conformal quantile `ceil((n + 1) * level) / n` (Vovk / Angelopoulos-Bates).
fn conformal_quantile(scores: &mut [f64], level: f64) -> f64 {
    if scores.is_empty() {
        return 0.5;
    }
    scores.sort_by(f64::total_cmp);
    let n = scores.len();
    let rank = (((n + 1) as f64) * level).ceil() as usize;
    let idx = rank.saturating_sub(1).min(n - 1);
    scores[idx]
}

/// Exact ROC AUC with tie-safe ordering.
#[must_use]
pub fn compute_roc_auc(scores: &[f64], labels: &[f64]) -> f64 {
    let mut pairs: Vec<(f64, bool)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(&s, &y)| (s, y > 0.5))
        .collect();
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    let n_pos = pairs.iter().filter(|x| x.1).count();
    let n_neg = pairs.len().saturating_sub(n_pos);
    if n_pos == 0 || n_neg == 0 {
        return 0.5;
    }
    let mut tp = 0.0f64;
    let mut auc = 0.0f64;
    let mut i = 0usize;
    while i < pairs.len() {
        let mut j = i + 1;
        while j < pairs.len() && (pairs[j].0 - pairs[i].0).abs() <= 1e-12 {
            j += 1;
        }
        let mut tie_pos = 0.0;
        let mut tie_neg = 0.0;
        for item in &pairs[i..j] {
            if item.1 {
                tie_pos += 1.0;
            } else {
                tie_neg += 1.0;
            }
        }
        auc += tie_neg * (tp + 0.5 * tie_pos);
        tp += tie_pos;
        i = j;
    }
    auc / ((n_pos as f64) * (n_neg as f64))
}

/// Average Precision (PR AUC).
#[must_use]
pub fn compute_pr_auc(scores: &[f64], labels: &[f64]) -> f64 {
    let mut pairs: Vec<(f64, bool)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(&s, &y)| (s, y > 0.5))
        .collect();
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    let n_pos = pairs.iter().filter(|x| x.1).count();
    if n_pos == 0 {
        return 0.0;
    }
    let mut tp = 0.0f64;
    let mut ap = 0.0f64;
    for (i, &(_, is_pos)) in pairs.iter().enumerate() {
        if is_pos {
            tp += 1.0;
            ap += tp / ((i + 1) as f64);
        }
    }
    ap / (n_pos as f64)
}

#[must_use]
pub fn compute_pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mx = xs[..n].iter().sum::<f64>() / nf;
    let my = ys[..n].iter().sum::<f64>() / nf;
    let mut num = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        num += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let den = (vx * vy).sqrt();
    if den > 1e-12 { num / den } else { 0.0 }
}

fn rank_vector(vals: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = vals.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut ranks = vec![0.0f64; vals.len()];
    let mut i = 0usize;
    while i < indexed.len() {
        let mut j = i + 1;
        while j < indexed.len() && (indexed[j].1 - indexed[i].1).abs() <= 1e-12 {
            j += 1;
        }
        let avg_rank = 0.5 * ((i + j - 1) as f64);
        for &(orig_idx, _) in &indexed[i..j] {
            ranks[orig_idx] = avg_rank;
        }
        i = j;
    }
    ranks
}

#[must_use]
pub fn compute_spearman(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return 0.0;
    }
    let rx = rank_vector(&xs[..n]);
    let ry = rank_vector(&ys[..n]);
    compute_pearson(&rx, &ry)
}
