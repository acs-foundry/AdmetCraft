#![forbid(unsafe_code)]

mod mcp;

use std::path::PathBuf;

use admet_calc::{MoleculeEvaluation, PkRoute, PkSimulationConfig, QsarTaskKind, TppPreset};
use admet_engine::{MAX_READ_BYTES, PathSandbox};
use admet_qsar::{
    AdmetModelPack, DEFAULT_MAX_METABOLITES, EvalOptions, GbdtTrainConfig, SeaOffTargetIndex,
    build_sea_index_from_tsv, compare_smiles_auto, default_model_pack_path, default_sea_index_path,
    dg_kcal_to_ki_nm, evaluate_batch_auto_with_options, evaluate_smiles_auto,
    load_default_model_pack, load_default_sea_index, predict_metabolism_sites,
    render_pk_dose_markdown, run_metabolite_guard, simulate_human_pk_dose, tdc_raw_dir,
    train_tdc_model_pack,
};
use admet_report::{
    ExportFormat, render_pk_profile_svg, write_report_file, write_report_to_writer,
};
use chem_core::{parse_csv_batch, parse_sdf, parse_smiles, parse_smiles_batch};
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum CliTpp {
    #[default]
    Oral,
    Cns,
    Peripheral,
    Oncology,
}

impl From<CliTpp> for TppPreset {
    fn from(value: CliTpp) -> Self {
        match value {
            CliTpp::Oral => TppPreset::OralSmallMolecule,
            CliTpp::Cns => TppPreset::CnsPenetrant,
            CliTpp::Peripheral => TppPreset::PeripheralSelective,
            CliTpp::Oncology => TppPreset::OncologyKinase,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum CliPkRoute {
    #[default]
    Oral,
    Iv,
    Sc,
}

impl From<CliPkRoute> for PkRoute {
    fn from(value: CliPkRoute) -> Self {
        match value {
            CliPkRoute::Oral => PkRoute::Oral,
            CliPkRoute::Iv => PkRoute::Iv,
            CliPkRoute::Sc => PkRoute::Sc,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "admetcraft-cli",
    version,
    about = "Local-first pure-Rust ADMET, TDC QSAR (90% Conformal CI + AD), SoM & TPP MPO calculator and MCP server"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Evaluate a single SMILES string and print complete ADMET + QSAR + SoM + TPP MPO JSON.
    Eval {
        /// SMILES string (or "SMILES Name").
        smiles: String,
        /// Optional molecule name.
        #[arg(long)]
        name: Option<String>,
        /// Target Product Profile preset for TPP MPO scoring.
        #[arg(long, value_enum, default_value = "oral")]
        tpp: CliTpp,
    },
    /// Print a concise human-readable ADMET, TDC QSAR (90% CI + AD), SoM & TPP MPO profile card.
    Profile {
        /// SMILES string (or "SMILES Name").
        smiles: String,
        /// Optional molecule name.
        #[arg(long)]
        name: Option<String>,
        /// Target Product Profile preset for TPP MPO scoring.
        #[arg(long, value_enum, default_value = "oral")]
        tpp: CliTpp,
    },
    /// Compare two molecules side-by-side (MMP / lead optimization delta JSON).
    Compare {
        /// Reference molecule SMILES (or "SMILES Name").
        smiles_a: String,
        /// Candidate analog SMILES (or "SMILES Name").
        smiles_b: String,
        /// Optional name for molecule A.
        #[arg(long)]
        name_a: Option<String>,
        /// Optional name for molecule B.
        #[arg(long)]
        name_b: Option<String>,
        /// Target Product Profile preset for TPP MPO scoring.
        #[arg(long, value_enum, default_value = "oral")]
        tpp: CliTpp,
    },
    /// Process a batch file (.smi, .csv, .sdf) in parallel across all CPU cores.
    Batch {
        /// Input file (.smi, .csv, .sdf).
        input: PathBuf,
        /// Output file path.
        #[arg(long)]
        out: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "csv")]
        format: CliFormat,
        /// Target Product Profile preset for TPP MPO scoring.
        #[arg(long, value_enum, default_value = "oral")]
        tpp: CliTpp,
        /// Skip Phase I/II metabolite re-screening for ultra-fast >10k compound screening.
        #[arg(long)]
        fast: bool,
        /// Optional read sandbox root.
        #[arg(long)]
        read_root: Option<PathBuf>,
        /// Optional write sandbox root.
        #[arg(long)]
        write_root: Option<PathBuf>,
        /// Explicitly allow creating a non-existent --write-root directory hierarchy.
        #[arg(long)]
        create_write_root: bool,
    },
    /// Train the local pure-Rust TDC QSAR GBDT ensemble (Bemis-Murcko scaffold split 70/10/20 + 90% conformal calibration).
    TrainQsar {
        /// Directory containing downloaded TDC `.tab` files (defaults to CRAFT_DATA_DIR/models/tdc_raw).
        #[arg(long)]
        tdc_dir: Option<PathBuf>,
        /// Output model pack path (defaults to CRAFT_DATA_DIR/models/admetcraft_v1/admetcraft_tdc_v1.json).
        #[arg(long)]
        out_pack: Option<PathBuf>,
        /// Optional Markdown validation report path.
        #[arg(long)]
        out_report: Option<PathBuf>,
        /// Maximum boosting trees per endpoint.
        #[arg(long, default_value_t = 240)]
        max_trees: usize,
        /// Maximum tree depth.
        #[arg(long, default_value_t = 5)]
        max_depth: usize,
    },
    /// Predict Phase I & Phase II hepatic/plasma metabolites and re-screen each for reactive bioactivation & toxicity escalation.
    Metabolites {
        /// SMILES string (or "SMILES Name").
        smiles: String,
        /// Optional molecule name.
        #[arg(long)]
        name: Option<String>,
        /// Maximum distinct metabolites to generate and re-screen.
        #[arg(long, default_value_t = DEFAULT_MAX_METABOLITES)]
        max_metabolites: usize,
        /// Emit full JSON instead of formatted summary table.
        #[arg(long)]
        json: bool,
    },
    /// Run local pure-Rust SEA (Similarity Ensemble Approach) off-target safety pharmacology fishing.
    Offtargets {
        /// SMILES string (or "SMILES Name").
        smiles: String,
        /// Optional molecule name.
        #[arg(long)]
        name: Option<String>,
        /// Optional custom SEA off-target index JSON (defaults to CRAFT_DATA_DIR or embedded Safety-20).
        #[arg(long)]
        sea_index: Option<PathBuf>,
        /// Emit full JSON instead of formatted summary table.
        #[arg(long)]
        json: bool,
    },
    /// Build a custom local SEA off-target index JSON from a TSV (`target_id\ttarget_name\ttarget_class\torgan_liability\tligand_name\tsmiles`).
    BuildSeaIndex {
        /// Input TSV path containing reference active ligands per target.
        #[arg(long)]
        input: PathBuf,
        /// Output JSON path (defaults to CRAFT_DATA_DIR/models/admetcraft_v1/sea_offtargets.json).
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Display held-out Bemis-Murcko scaffold validation metrics for the installed local QSAR model pack.
    ModelInfo {
        /// Optional explicit model pack path (defaults to CRAFT_DATA_DIR/models/admetcraft_v1/admetcraft_tdc_v1.json).
        #[arg(long)]
        model_pack: Option<PathBuf>,
    },
    /// Mechanistic IVIVE -> Minimal-PBPK human pharmacokinetics & therapeutic dose prediction (QD/BID + 90% conformal CI + SVG curve).
    PkDose {
        /// SMILES string (or "SMILES Name").
        smiles: String,
        /// Optional molecule name.
        #[arg(long)]
        name: Option<String>,
        /// Target biochemical/cellular potency (`IC50` / `Ki`) in nM (overridden if `--dg-kcal` is passed).
        #[arg(long, default_value_t = 10.0)]
        potency_nm: f64,
        /// Optional binding free energy `ΔG` in kcal/mol (converted to `Ki` nM at 298.15 K).
        #[arg(long, allow_hyphen_values = true)]
        dg_kcal: Option<f64>,
        /// Required unbound concentration multiple of `IC50`/`Ki` (default `1.0`).
        #[arg(long, default_value_t = 1.0)]
        fold: f64,
        /// Match steady-state average unbound concentration (`Cavg,u`) instead of trough (`Cmin,u`).
        #[arg(long)]
        match_average: bool,
        /// Dosing route (`oral` | `iv` | `sc`).
        #[arg(long, value_enum, default_value = "oral")]
        route: CliPkRoute,
        /// Adult body weight in kg (default `70.0`).
        #[arg(long, default_value_t = 70.0)]
        body_weight: f64,
        /// Target Product Profile preset for baseline evaluation.
        #[arg(long, value_enum, default_value = "oral")]
        tpp: CliTpp,
        /// Optional JSON output path for complete `PbpkHumanDoseReport`.
        #[arg(long)]
        out_json: Option<PathBuf>,
        /// Optional standalone SVG output path for steady-state `C_u(t)` profile with 90% CI ribbon.
        #[arg(long)]
        out_svg: Option<PathBuf>,
        /// Optional Markdown report output path.
        #[arg(long)]
        out_md: Option<PathBuf>,
        /// Print raw JSON on stdout instead of Markdown summary.
        #[arg(long)]
        json: bool,
    },
    /// Start the Model Context Protocol (MCP) server on stdio or TCP.
    Mcp {
        /// Authorized directory for reading input files.
        #[arg(long)]
        read_root: PathBuf,
        /// Authorized directory for writing output reports.
        #[arg(long)]
        write_root: PathBuf,
        /// Explicitly allow creating a non-existent --write-root directory hierarchy.
        #[arg(long)]
        create_write_root: bool,
        /// Optional TCP bridge address (loopback by default; requires --mcp-token).
        #[arg(long)]
        tcp: Option<String>,
        /// Allow binding non-loopback IP addresses for --tcp (must also provide --mcp-token).
        #[arg(long)]
        allow_remote: bool,
        /// Bearer token required by --tcp clients during `initialize` (or env ADMETCRAFT_MCP_TOKEN).
        #[arg(long)]
        mcp_token: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliFormat {
    Csv,
    Json,
    Pdf,
    Svg,
}

impl From<CliFormat> for ExportFormat {
    fn from(value: CliFormat) -> Self {
        match value {
            CliFormat::Csv => ExportFormat::Csv,
            CliFormat::Json => ExportFormat::Json,
            CliFormat::Pdf => ExportFormat::Pdf,
            CliFormat::Svg => ExportFormat::SvgBoiledEgg,
        }
    }
}

fn split_smiles_and_name<'a>(
    raw: &'a str,
    explicit: Option<&'a str>,
) -> (&'a str, Option<&'a str>) {
    let mut parts = raw.trim().splitn(2, char::is_whitespace);
    let smi_token = parts.next().unwrap_or("");
    let inline_name = parts.next().map(str::trim).filter(|s| !s.is_empty());
    (smi_token, explicit.or(inline_name))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Eval { smiles, name, tpp } => {
            let (smi_token, final_name) = split_smiles_and_name(&smiles, name.as_deref());
            let eval = evaluate_smiles_auto(smi_token, final_name, tpp.into())?;
            let out = serde_json::to_string_pretty(&eval)?;
            println!("{out}");
        }
        Commands::Profile { smiles, name, tpp } => {
            let (smi_token, final_name) = split_smiles_and_name(&smiles, name.as_deref());
            let eval = evaluate_smiles_auto(smi_token, final_name, tpp.into())?;
            print_profile(&eval);
        }
        Commands::Compare {
            smiles_a,
            smiles_b,
            name_a,
            name_b,
            tpp,
        } => {
            let (smi_a, final_a) = split_smiles_and_name(&smiles_a, name_a.as_deref());
            let (smi_b, final_b) = split_smiles_and_name(&smiles_b, name_b.as_deref());
            let cmp = compare_smiles_auto(smi_a, final_a, smi_b, final_b, tpp.into())?;
            let out = serde_json::to_string_pretty(&cmp)?;
            println!("{out}");
        }
        Commands::Batch {
            input,
            out,
            format,
            tpp,
            fast,
            read_root,
            write_root,
            create_write_root,
        } => {
            if read_root.is_none() || write_root.is_none() {
                eprintln!(
                    "Notice: running batch without full --read-root/--write-root sandbox (direct local filesystem access)."
                );
            }
            let (ext, content) = if let Some(rr) = read_root.as_deref() {
                let sb = PathSandbox::new_with_options(Some(rr), None, false)?;
                let (resolved_in, text) = sb.read_to_string_limited(&input, MAX_READ_BYTES)?;
                let ext = resolved_in
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                (ext, text)
            } else {
                let ext = input
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                (ext, std::fs::read_to_string(&input)?)
            };
            let opts = EvalOptions {
                run_offtargets: true,
                run_metabolites: !fast,
                max_metabolites: DEFAULT_MAX_METABOLITES,
            };
            let evals = load_molecules_from_text(&ext, &content, tpp.into(), opts)?;
            if let Some(wr) = write_root.as_deref() {
                let sb = PathSandbox::new_with_options(None, Some(wr), create_write_root)?;
                let (resolved_out, mut file) = sb.open_write(&out)?;
                write_report_to_writer(&evals, &mut file, format.into())?;
                eprintln!(
                    "Exported {} molecule(s) to {}",
                    evals.len(),
                    resolved_out.display()
                );
            } else {
                write_report_file(&evals, &out, format.into())?;
                eprintln!("Exported {} molecule(s) to {}", evals.len(), out.display());
            }
        }
        Commands::TrainQsar {
            tdc_dir,
            out_pack,
            out_report,
            max_trees,
            max_depth,
        } => {
            let src = tdc_dir.unwrap_or_else(tdc_raw_dir);
            let dst = out_pack.unwrap_or_else(default_model_pack_path);
            let cfg = GbdtTrainConfig {
                max_trees: max_trees.clamp(20, 600),
                max_depth: max_depth.clamp(2, 7),
                ..Default::default()
            };
            let pack = train_tdc_model_pack(&src, &dst, out_report.as_deref(), &cfg)?;
            println!("{}", pack.render_validation_markdown());
            eprintln!(
                "Saved {} trained QSAR endpoint(s) -> {}",
                pack.models.len(),
                dst.display()
            );
        }
        Commands::Metabolites {
            smiles,
            name,
            max_metabolites,
            json,
        } => {
            let (smi_token, final_name) = split_smiles_and_name(&smiles, name.as_deref());
            let mut mol = parse_smiles(smi_token)?;
            if let Some(n) = final_name {
                mol.name = Some(n.to_string());
            }
            let pack = load_default_model_pack();
            let mut parent_eval = admet_calc::evaluate_molecule(&mol);
            if let Some(mp) = pack {
                let feat = admet_qsar::extract_mol_features(&mol, &parent_eval);
                parent_eval.qsar = Some(mp.predict_all(&feat, "local"));
            }
            let sites = predict_metabolism_sites(&mol);
            let report = run_metabolite_guard(&mol, &parent_eval, &sites, pack, max_metabolites);
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "=== Metabolite Guard: {} ({} generated | {} bioactivation | {} escalation) ===",
                    parent_eval.name,
                    report.total_generated,
                    report.bioactivation_alerts,
                    report.escalation_count
                );
                for m in &report.metabolites {
                    let flag = if m.reactive_bioactivation {
                        "BIOACTIVATION"
                    } else if m.toxicity_escalation {
                        "ESCALATION"
                    } else {
                        "OK"
                    };
                    println!(
                        "  #{:<2} [{}] {:8} {:42} | DILI={} AMES={} | {}",
                        m.rank,
                        flag,
                        m.phase.as_str(),
                        m.reaction,
                        m.dili_prob
                            .map_or_else(|| "-".to_string(), |v| format!("{v:.2}")),
                        m.ames_prob
                            .map_or_else(|| "-".to_string(), |v| format!("{v:.2}")),
                        m.smiles
                    );
                    if let Some(reason) = &m.bioactivation_reason {
                        println!("      -> {reason}");
                    }
                }
            }
        }
        Commands::Offtargets {
            smiles,
            name,
            sea_index,
            json,
        } => {
            let (smi_token, final_name) = split_smiles_and_name(&smiles, name.as_deref());
            let mut mol = parse_smiles(smi_token)?;
            if let Some(n) = final_name {
                mol.name = Some(n.to_string());
            }
            let report = if let Some(path) = sea_index {
                let idx = SeaOffTargetIndex::load_from_path(&path)?;
                idx.predict_molecule(&mol)
            } else {
                load_default_sea_index().predict_molecule(&mol)
            };
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "=== SEA Off-Target Profile: {} ({} | {} targets | {} High, {} Moderate) ===",
                    mol.name.as_deref().unwrap_or("Query"),
                    report.panel_version,
                    report.targets_evaluated,
                    report.high_risk_count,
                    report.moderate_risk_count
                );
                if report.hits.is_empty() {
                    println!("  No significant off-target hits (all Low risk).");
                } else {
                    for h in &report.hits {
                        println!(
                            "  {:14} [{:8}] maxTc={:.3} Z={:+6.2} E={:.2e} | nearest={:16} | {}",
                            h.target_id,
                            h.risk.as_str(),
                            h.max_tanimoto,
                            h.z_score,
                            h.e_value,
                            h.nearest_ligand,
                            h.organ_liability
                        );
                    }
                }
            }
        }
        Commands::BuildSeaIndex { input, out } => {
            let dst = out.unwrap_or_else(default_sea_index_path);
            let idx = build_sea_index_from_tsv(&input, &dst)?;
            let total_ligs: usize = idx.targets.iter().map(|t| t.ligands.len()).sum();
            println!(
                "Built SEA index ({} targets, {} ligands) -> {}",
                idx.targets.len(),
                total_ligs,
                dst.display()
            );
        }
        Commands::ModelInfo { model_pack } => {
            if let Some(path) = model_pack {
                let pack = AdmetModelPack::load_from_path(&path)?;
                println!("{}", pack.render_validation_markdown());
            } else if let Some(pack) = load_default_model_pack() {
                println!("{}", pack.render_validation_markdown());
            } else {
                eprintln!(
                    "No local TDC QSAR model pack found at {}. Run `admetcraft-cli train-qsar` first.",
                    default_model_pack_path().display()
                );
            }
        }
        Commands::PkDose {
            smiles,
            name,
            potency_nm,
            dg_kcal,
            fold,
            match_average,
            route,
            body_weight,
            tpp,
            out_json,
            out_svg,
            out_md,
            json,
        } => {
            let (smi, final_name) = split_smiles_and_name(&smiles, name.as_deref());
            let eval = evaluate_smiles_auto(smi, final_name, tpp.into())?;
            let eff_potency = dg_kcal.map_or(potency_nm.max(0.01), dg_kcal_to_ki_nm);
            let cfg = PkSimulationConfig {
                target_potency_nm: eff_potency,
                target_fold: fold.max(0.1),
                match_trough: !match_average,
                route: route.into(),
                body_weight_kg: body_weight.clamp(10.0, 200.0),
                ..Default::default()
            };
            let report = simulate_human_pk_dose(&eval, &cfg);
            if let Some(jp) = &out_json {
                if let Some(par) = jp.parent()
                    && !par.as_os_str().is_empty()
                {
                    std::fs::create_dir_all(par)?;
                }
                std::fs::write(jp, serde_json::to_string_pretty(&report)?)?;
            }
            if let Some(sp) = &out_svg {
                if let Some(par) = sp.parent()
                    && !par.as_os_str().is_empty()
                {
                    std::fs::create_dir_all(par)?;
                }
                std::fs::write(sp, render_pk_profile_svg(&report))?;
            }
            let md = render_pk_dose_markdown(&report);
            if let Some(mp) = &out_md {
                if let Some(par) = mp.parent()
                    && !par.as_os_str().is_empty()
                {
                    std::fs::create_dir_all(par)?;
                }
                std::fs::write(mp, &md)?;
            }
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!("{md}");
            }
        }
        Commands::Mcp {
            read_root,
            write_root,
            create_write_root,
            tcp,
            allow_remote,
            mcp_token,
        } => {
            let sandbox = PathSandbox::new_with_options(
                Some(&read_root),
                Some(&write_root),
                create_write_root,
            )?;
            if let Some(addr) = tcp {
                let token = mcp_token.or_else(|| std::env::var("ADMETCRAFT_MCP_TOKEN").ok());
                mcp::run_mcp_tcp(&addr, &sandbox, allow_remote, token)?;
            } else {
                mcp::run_mcp_stdio(&sandbox)?;
            }
        }
    }
    Ok(())
}

fn yes_no(b: bool) -> &'static str {
    if b { "Yes" } else { "No" }
}

fn print_profile(e: &MoleculeEvaluation) {
    let p = &e.physicochemical;
    let l = &e.lipophilicity;
    let ion = &e.ionization;
    let w = &e.water_solubility;
    let pk = &e.pharmacokinetics;
    let d = &e.druglikeness;
    let mc = &e.medicinal_chemistry;
    let s = &e.safety;

    println!("=== {} ({}) ===", e.name, p.formula);
    println!("SMILES        : {}", e.canonical_smiles);
    println!(
        "Physicochem   : MW={:.2} Da | Exact={:.4} Da | HA={} (Ar={}) | Rings={} (Ar={}) | DBE={:.1}",
        p.molecular_weight,
        p.exact_mass,
        p.heavy_atoms,
        p.aromatic_heavy_atoms,
        p.num_rings,
        p.num_aromatic_rings,
        p.dbe
    );
    println!(
        "Polarity/Flex : TPSA={:.2} A^2 (N/O={:.2}) | HBA={} | HBD={} | RotB={} | Fsp3={:.2} | MR={:.2}",
        p.tpsa,
        p.tpsa_no_only,
        p.h_bond_acceptors,
        p.h_bond_donors,
        p.rotatable_bonds,
        p.fraction_csp3,
        p.molar_refractivity
    );
    println!(
        "Lipophilicity : WLOGP={:.2} | XLOGP3={:.2} | MLOGP={:.2} | Consensus={:.2}",
        l.wlogp, l.xlogp3, l.mlogp, l.consensus_log_p
    );
    let pka_a = ion
        .strongest_acidic_pka
        .map(|v| format!("{v:.1}"))
        .unwrap_or_else(|| "-".to_string());
    let pka_b = ion
        .strongest_basic_pka
        .map(|v| format!("{v:.1}"))
        .unwrap_or_else(|| "-".to_string());
    println!(
        "Ionization    : {} | LogD7.4={:.2} | Charge@7.4={:+.2} | pKa(a)={} pKa(b)={}",
        ion.ionization_class.as_str(),
        ion.logd_ph74,
        ion.estimated_charge_ph74,
        pka_a,
        pka_b
    );
    println!(
        "Solubility    : ESOL={:.2} ({}) | Ali={:.2} ({})",
        w.esol.log_s,
        w.esol.class.as_str(),
        w.ali.log_s,
        w.ali.class.as_str()
    );
    println!(
        "PK/BOILED-Egg : GI={} | BBB={} | LogKp={:.2} cm/s | P-gp(exp)={} | CYP(exp) 1A2:{} 2C19:{} 2C9:{} 2D6:{} 3A4:{}",
        pk.gi_absorption.as_str(),
        yes_no(pk.bbb_permeant),
        pk.log_kp_cm_s,
        yes_no(pk.pgp_substrate),
        yes_no(pk.cyp1a2_inhibitor),
        yes_no(pk.cyp2c19_inhibitor),
        yes_no(pk.cyp2c9_inhibitor),
        yes_no(pk.cyp2d6_inhibitor),
        yes_no(pk.cyp3a4_inhibitor)
    );
    println!(
        "Druglikeness  : QED={:.3} | CNS_MPO={:.2}/6.0 | Bioavail={:.2} | Lipinski={}({}) | Veber={} | Ghose={} | Egan={} | Muegge={}",
        d.qed,
        d.cns_mpo,
        d.bioavailability_score,
        yes_no(d.lipinski.pass),
        d.lipinski.violations,
        yes_no(d.veber.pass),
        yes_no(d.ghose.pass),
        yes_no(d.egan.pass),
        yes_no(d.muegge.pass)
    );
    println!(
        "MedChem Rules : Pfizer3/75={} | GSK4/400={} | GoldenTri={} | Ro3={} | bRo5={} | Leadlike={} | SA(proxy)={:.2}",
        yes_no(d.pfizer_3_75.pass),
        yes_no(d.gsk_4_400.pass),
        yes_no(d.golden_triangle.pass),
        yes_no(d.ro3_fragment.pass),
        yes_no(d.beyond_ro5),
        yes_no(mc.leadlikeness.pass),
        mc.synthetic_accessibility
    );
    let fmt_alerts = |xs: &[String]| -> String {
        if xs.is_empty() {
            "none".to_string()
        } else {
            xs.join(", ")
        }
    };
    println!(
        "Alerts/Safety : PAINS({}/{})=[{}] | Brenk({}/{})=[{}]",
        mc.pains_alerts.len(),
        mc.pains_catalog_size,
        fmt_alerts(&mc.pains_alerts),
        mc.brenk_alerts.len(),
        mc.brenk_catalog_size,
        fmt_alerts(&mc.brenk_alerts)
    );
    println!(
        "Toxicity(exp) : hERG={} | AMES({})=[{}] | DILI({})=[{}]",
        s.herg_risk.as_str(),
        s.ames_alerts.len(),
        fmt_alerts(&s.ames_alerts),
        s.dili_alerts.len(),
        fmt_alerts(&s.dili_alerts)
    );
    if let Some(tpp) = &e.tpp_mpo {
        println!(
            "TPP MPO       : {:.3} ({}) [{}]",
            tpp.score,
            tpp.decision,
            tpp.preset.as_str()
        );
        for hs in &tpp.hard_stops {
            println!("  HARD STOP   : {hs}");
        }
    }
    if let Some(qsar) = &e.qsar {
        println!(
            "QSAR Panel    : {} endpoints (v{}) | AD={} (maxSim={:.3}, top5={:.3}, Z={:.2})",
            qsar.endpoints.len(),
            qsar.model_pack_version,
            qsar.overall_ad.status.as_str(),
            qsar.overall_ad.max_tanimoto,
            qsar.overall_ad.top5_tanimoto,
            qsar.overall_ad.descriptor_z_rms
        );
        for ep in &qsar.endpoints {
            match ep.task {
                QsarTaskKind::BinaryClassification => {
                    println!(
                        "  {:25} P={:.3} [{:.2}..{:.2}] | {:19} | AD={:13} ({}={:.3})",
                        ep.endpoint_id,
                        ep.value,
                        ep.ci90_low,
                        ep.ci90_high,
                        ep.conformal_call,
                        ep.applicability_domain.status.as_str(),
                        ep.test_metric_name,
                        ep.test_metric_value
                    );
                }
                QsarTaskKind::Regression => {
                    println!(
                        "  {:25} {:34} | AD={:13} ({}={:.3})",
                        ep.endpoint_id,
                        ep.conformal_call,
                        ep.applicability_domain.status.as_str(),
                        ep.test_metric_name,
                        ep.test_metric_value
                    );
                }
            }
        }
    }
    if !e.metabolism_sites.is_empty() {
        println!("Sites of Met. : {} hotspot(s)", e.metabolism_sites.len());
        for s_m in e.metabolism_sites.iter().take(5) {
            println!(
                "  #{:<2} {:2} {:18} | {:36} | {:.2} ({}) -> {}",
                s_m.atom_index + 1,
                s_m.atom_symbol,
                s_m.isoform,
                s_m.reaction,
                s_m.lability_score,
                s_m.risk.as_str(),
                s_m.medchem_fix
            );
        }
    }
    if let Some(tox21) = &e.tox21_summary {
        let tot = tox21.endocrine_alerts.len() + tox21.stress_alerts.len();
        println!(
            "Tox21 Panel   : {} assays | {} alert(s) (NR={}, SR={}, max P={:.2})",
            tox21.assays_evaluated,
            tot,
            tox21.endocrine_alerts.len(),
            tox21.stress_alerts.len(),
            tox21.max_probability
        );
        for a in &tox21.endocrine_alerts {
            println!("  NR ALERT    : {a}");
        }
        for a in &tox21.stress_alerts {
            println!("  SR ALERT    : {a}");
        }
    }
    if let Some(sea) = &e.off_targets {
        println!(
            "SEA OffTarget : {} ({} targets | {} High, {} Moderate)",
            sea.panel_version, sea.targets_evaluated, sea.high_risk_count, sea.moderate_risk_count
        );
        for h in sea.hits.iter().take(5) {
            println!(
                "  {:14} [{:8}] maxTc={:.3} Z={:+5.2} ({}) -> {}",
                h.target_id,
                h.risk.as_str(),
                h.max_tanimoto,
                h.z_score,
                h.nearest_ligand,
                h.organ_liability
            );
        }
    }
    if let Some(mg) = &e.metabolite_guard {
        println!(
            "Metab. Guard  : {} generated | {} bioactivation | {} escalation",
            mg.total_generated, mg.bioactivation_alerts, mg.escalation_count
        );
        for m in mg.metabolites.iter().take(5) {
            let tag = if m.reactive_bioactivation {
                "BIOACT"
            } else if m.toxicity_escalation {
                "ESC"
            } else {
                "OK"
            };
            println!(
                "  #{:<2} [{:6}] {:7} {:40} | {}",
                m.rank,
                tag,
                m.phase.as_str(),
                m.reaction,
                m.smiles
            );
        }
    }
    for w_msg in &e.warnings {
        println!("Warning       : {w_msg}");
    }
}

fn load_molecules_from_text(
    ext: &str,
    content: &str,
    tpp: TppPreset,
    opts: EvalOptions,
) -> Result<Vec<admet_calc::MoleculeEvaluation>, Box<dyn std::error::Error>> {
    let mut mols = Vec::new();
    if ext == "sdf" || ext == "mol" {
        for (i, res) in parse_sdf(content).into_iter().enumerate() {
            match res {
                Ok(m) => mols.push(m),
                Err(e) => eprintln!("Warning: SDF record {} skipped: {e}", i + 1),
            }
        }
    } else if ext == "csv" {
        for (i, res) in parse_csv_batch(content)?.into_iter().enumerate() {
            match res {
                Ok(m) => mols.push(m),
                Err(e) => eprintln!("Warning: CSV row {} skipped: {e}", i + 2),
            }
        }
    } else {
        for (line, _, res) in parse_smiles_batch(content) {
            match res {
                Ok(m) => mols.push(m),
                Err(e) => eprintln!("Warning: line {line} skipped: {e}"),
            }
        }
    }
    Ok(evaluate_batch_auto_with_options(&mols, tpp, opts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;

    #[test]
    fn mcp_server_handles_evaluate_and_rejects_outside_sandbox()
    -> Result<(), Box<dyn std::error::Error>> {
        let base = std::env::temp_dir().join("admetcraft_mcp_cli_test_p0");
        let _ = std::fs::remove_dir_all(&base);
        let rdir = base.join("read");
        let wdir = base.join("write");
        let forbidden = base.join("forbidden");
        std::fs::create_dir_all(&rdir)?;
        std::fs::create_dir_all(&wdir)?;
        std::fs::create_dir_all(&forbidden)?;
        std::fs::write(forbidden.join("secret.smi"), "CCO Secret\n")?;

        let sb = PathSandbox::new(Some(&rdir), Some(&wdir))?;
        let requests = [
            serde_json::json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}).to_string(),
            serde_json::json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"evaluate_smiles","arguments":{"smiles":"CC(=O)Oc1ccccc1C(=O)O Aspirin"}}}).to_string(),
            serde_json::json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"evaluate_smiles","arguments":{"file":"../forbidden/secret.smi"}}}).to_string(),
            serde_json::json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"export_report","arguments":{"smiles":["CCO Ethanol"],"output_path":"../forbidden/evil.csv","format":"csv"}}}).to_string(),
            serde_json::json!({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"export_report","arguments":{"smiles":["CCO Ethanol"],"output_path":"report.csv","format":"csv"}}}).to_string(),
        ].join("\n") + "\n";

        let reader = BufReader::new(requests.as_bytes());
        let mut out_buf = Vec::new();
        mcp::serve_stream_for_test(reader, &mut out_buf, &sb, None)?;
        let out_str = String::from_utf8(out_buf)?;
        let lines: Vec<serde_json::Value> = out_str
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?;
        assert_eq!(lines.len(), 5);
        assert_eq!(lines[1]["result"]["isError"], false);
        assert_eq!(lines[1]["result"]["structuredContent"]["count"], 1);
        assert_eq!(lines[2]["result"]["isError"], true);
        assert_eq!(lines[3]["result"]["isError"], true);
        assert_eq!(lines[4]["result"]["isError"], false);
        assert!(wdir.join("report.csv").exists());

        let _ = std::fs::remove_dir_all(&base);
        Ok(())
    }

    #[test]
    fn mcp_strict_jsonrpc20_compliance_and_compare_tool() -> Result<(), Box<dyn std::error::Error>>
    {
        let base = std::env::temp_dir().join("admetcraft_mcp_jsonrpc20_test");
        let _ = std::fs::remove_dir_all(&base);
        let rdir = base.join("read");
        let wdir = base.join("write");
        std::fs::create_dir_all(&rdir)?;
        std::fs::create_dir_all(&wdir)?;
        let sb = PathSandbox::new_with_options(Some(&rdir), Some(&wdir), false)?;

        // Missing write_root must be rejected when create_write_root is false
        assert!(
            PathSandbox::new_with_options(Some(&rdir), Some(&base.join("nonexistent_dir")), false)
                .is_err()
        );

        let requests = [
            // 1. Missing jsonrpc -> -32600
            serde_json::json!({"id":1,"method":"ping"}).to_string(),
            // 2. Wrong jsonrpc "1.0" -> -32600
            serde_json::json!({"jsonrpc":"1.0","id":2,"method":"ping"}).to_string(),
            // 3. Explicit id: null -> MUST reply with id: null
            serde_json::json!({"jsonrpc":"2.0","id":null,"method":"ping"}).to_string(),
            // 4. Initialize
            serde_json::json!({"jsonrpc":"2.0","id":3,"method":"initialize","params":{}}).to_string(),
            // 5. Notification (no 'id' member) -> MUST produce NO output
            serde_json::json!({"jsonrpc":"2.0","method":"notifications/initialized"}).to_string(),
            // 6. tools/list -> all 5 tools must expose outputSchema
            serde_json::json!({"jsonrpc":"2.0","id":4,"method":"tools/list"}).to_string(),
            // 7. Negative max_lipinski_violations -> -32602 Invalid params
            serde_json::json!({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"filter_druglikeness","arguments":{"smiles":["CCO"],"max_lipinski_violations":-1}}}).to_string(),
            // 8. export_report missing required 'format' -> -32602 (no silent fallback to csv)
            serde_json::json!({"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"export_report","arguments":{"smiles":["CCO"],"output_path":"x.csv"}}}).to_string(),
            // 9. export_report unsupported 'format' -> -32602
            serde_json::json!({"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"export_report","arguments":{"smiles":["CCO"],"output_path":"x.xml","format":"xml"}}}).to_string(),
            // 10. compare_molecules tool
            serde_json::json!({"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"compare_molecules","arguments":{"smiles_a":"CC(C)Cc1ccc(C(C)C(=O)O)cc1","smiles_b":"CC(C)Cc1ccc(C(C)c2nnn[nH]2)cc1","name_a":"Ibuprofen","name_b":"Ibu_Tetrazole"}}}).to_string(),
        ].join("\n") + "\n";

        let mut out_buf = Vec::new();
        mcp::serve_stream_for_test(BufReader::new(requests.as_bytes()), &mut out_buf, &sb, None)?;
        let out_str = String::from_utf8(out_buf)?;
        let lines: Vec<serde_json::Value> = out_str
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?;

        // 10 inputs minus 1 notification = 9 responses
        assert_eq!(lines.len(), 9);
        assert_eq!(lines[0]["error"]["code"], -32600);
        assert_eq!(lines[1]["error"]["code"], -32600);
        assert!(lines[2]["id"].is_null());
        assert_eq!(lines[2]["result"], serde_json::json!({}));
        assert_eq!(lines[3]["id"], 3);

        let tools = lines[4]["result"]["tools"]
            .as_array()
            .ok_or("tools must be array")?;
        assert_eq!(tools.len(), 9);
        for t in tools {
            assert!(
                t.get("outputSchema").is_some(),
                "tool {} missing outputSchema",
                t["name"]
            );
        }

        assert_eq!(lines[5]["error"]["code"], -32602);
        assert_eq!(lines[6]["error"]["code"], -32602);
        assert_eq!(lines[7]["error"]["code"], -32602);
        assert_eq!(lines[8]["result"]["isError"], false);
        assert_eq!(
            lines[8]["result"]["structuredContent"]["name_a"],
            "Ibuprofen"
        );
        assert!(
            lines[8]["result"]["structuredContent"]["delta_tpsa"]
                .as_f64()
                .unwrap_or(0.0)
                > 15.0
        );

        let _ = std::fs::remove_dir_all(&base);
        Ok(())
    }

    #[test]
    fn mcp_enforces_auth_initialize_size_and_batch_limits() -> Result<(), Box<dyn std::error::Error>>
    {
        let base = std::env::temp_dir().join("admetcraft_mcp_limits_test");
        let _ = std::fs::remove_dir_all(&base);
        let rdir = base.join("read");
        let wdir = base.join("write");
        std::fs::create_dir_all(&rdir)?;
        std::fs::create_dir_all(&wdir)?;
        let sb = PathSandbox::new(Some(&rdir), Some(&wdir))?;

        // 1. Uninitialized call + wrong token must both be rejected
        let uninit_reqs = [
            serde_json::json!({"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}).to_string(),
            serde_json::json!({"jsonrpc":"2.0","id":2,"method":"initialize","params":{"token":"wrong"}}).to_string(),
            serde_json::json!({"jsonrpc":"2.0","id":3,"method":"initialize","params":{"token":"s3cr3t"}}).to_string(),
            serde_json::json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"evaluate_smiles","arguments":{"smiles": vec!["CCO"; 1005]}}}).to_string(),
        ].join("\n") + "\n";

        let mut out_buf = Vec::new();
        mcp::serve_stream_for_test(
            BufReader::new(uninit_reqs.as_bytes()),
            &mut out_buf,
            &sb,
            Some("s3cr3t".to_string()),
        )?;
        let out_str = String::from_utf8(out_buf)?;
        let lines: Vec<serde_json::Value> = out_str
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?;
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0]["error"]["code"], -32002);
        assert_eq!(lines[1]["error"]["code"], -32001);
        assert!(lines[2]["result"]["serverInfo"]["name"] == "admetcraft-mcp");
        assert_eq!(lines[3]["result"]["isError"], true);

        // 2. Oversized >1 MiB line must be rejected with -32600
        let huge_line = "x".repeat(mcp::MAX_MCP_MESSAGE_BYTES + 128) + "\n";
        let mut huge_out = Vec::new();
        mcp::serve_stream_for_test(
            BufReader::new(huge_line.as_bytes()),
            &mut huge_out,
            &sb,
            None,
        )?;
        let resp: serde_json::Value = serde_json::from_str(String::from_utf8(huge_out)?.trim())?;
        assert_eq!(resp["error"]["code"], -32600);

        // 3. Non-loopback TCP without --allow-remote must be refused
        assert!(mcp::run_mcp_tcp("0.0.0.0:9999", &sb, false, Some("tok".to_string())).is_err());

        let _ = std::fs::remove_dir_all(&base);
        Ok(())
    }

    #[test]
    fn mcp_tcp_handles_concurrent_clients_despite_idle_socket_and_catches_panics()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::io::{BufRead, BufReader, Write};
        use std::net::{TcpListener, TcpStream};
        use std::sync::{Arc, atomic::AtomicUsize};
        use std::time::Duration;

        let base = std::env::temp_dir().join("admetcraft_mcp_tcp_concurrency_test");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base)?;
        let sb = Arc::new(PathSandbox::new(Some(&base), Some(&base))?);

        let listener = TcpListener::bind("127.0.0.1:0")?;
        let addr = listener.local_addr()?;
        let active = Arc::new(AtomicUsize::new(0));
        let sb_srv = Arc::clone(&sb);
        let active_srv = Arc::clone(&active);

        let server_thread = std::thread::spawn(move || {
            for _ in 0..2 {
                if let Ok((stream, _)) = listener.accept() {
                    mcp::handle_tcp_stream_for_test(
                        stream,
                        Arc::clone(&sb_srv),
                        "tok123".to_string(),
                        Arc::clone(&active_srv),
                        Duration::from_secs(2),
                    );
                }
            }
        });

        // Client 1 stays idle without sending bytes
        let _idle_client = TcpStream::connect(addr)?;

        // Client 2 initializes and runs evaluate_smiles concurrently without waiting on Client 1
        let mut active_client = TcpStream::connect(addr)?;
        active_client.set_read_timeout(Some(Duration::from_secs(2)))?;
        let init_req = serde_json::json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"token":"tok123"}}).to_string() + "\n";
        let eval_req = serde_json::json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"evaluate_smiles","arguments":{"smiles":"CCO Ethanol"}}}).to_string() + "\n";
        active_client.write_all(init_req.as_bytes())?;
        active_client.write_all(eval_req.as_bytes())?;
        active_client.flush()?;

        let mut reader = BufReader::new(active_client);
        let mut line1 = String::new();
        let mut line2 = String::new();
        reader.read_line(&mut line1)?;
        reader.read_line(&mut line2)?;
        let v1: serde_json::Value = serde_json::from_str(line1.trim())?;
        let v2: serde_json::Value = serde_json::from_str(line2.trim())?;
        assert_eq!(v1["id"], 1);
        assert_eq!(v2["result"]["isError"], false);
        assert_eq!(v2["result"]["structuredContent"]["count"], 1);

        let _ = server_thread.join();

        // Panic isolation check
        #[allow(clippy::panic)]
        let isolated = mcp::run_isolated_for_test(serde_json::json!(99), || {
            panic!("simulated request panic");
        })
        .ok_or("missing serialized panic response")?;
        let isolated_v: serde_json::Value = serde_json::from_str(&isolated)?;
        assert_eq!(isolated_v["error"]["code"], -32603);

        let _ = std::fs::remove_dir_all(&base);
        Ok(())
    }

    #[test]
    fn mcp_predict_metabolites_and_offtargets_tools_work() -> Result<(), Box<dyn std::error::Error>>
    {
        let base = std::env::temp_dir().join("admetcraft_mcp_ola1_test");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base)?;
        let sb = PathSandbox::new(Some(&base), Some(&base))?;

        let requests = [
            serde_json::json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}).to_string(),
            serde_json::json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"predict_metabolites","arguments":{"smiles":"CC(=O)Nc1ccc(O)cc1","name":"Paracetamol"}}}).to_string(),
            serde_json::json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"predict_offtargets","arguments":{"smiles":"O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1","name":"Haloperidol"}}}).to_string(),
            serde_json::json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"predict_human_pk_dose","arguments":{"smiles":"CC(C)Cc1ccc(C(C)C(=O)O)cc1","name":"Ibuprofen","potency_nm":50.0,"out_svg_path":"ibu_pk.svg"}}}).to_string(),
            serde_json::json!({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"predict_human_pk_dose","arguments":{"smiles":"CCO","out_svg_path":"../escape.svg"}}}).to_string(),
        ].join("\n") + "\n";

        let mut out_buf = Vec::new();
        mcp::serve_stream_for_test(BufReader::new(requests.as_bytes()), &mut out_buf, &sb, None)?;
        let out_str = String::from_utf8(out_buf)?;
        let lines: Vec<serde_json::Value> = out_str
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?;
        assert_eq!(lines.len(), 5);
        assert_eq!(lines[1]["result"]["isError"], false);
        assert!(
            lines[1]["result"]["structuredContent"]["bioactivation_alerts"]
                .as_u64()
                .unwrap_or(0)
                >= 1
        );
        assert_eq!(lines[2]["result"]["isError"], false);
        assert!(
            lines[2]["result"]["structuredContent"]["high_risk_count"]
                .as_u64()
                .unwrap_or(0)
                >= 1
        );
        assert_eq!(lines[3]["result"]["isError"], false);
        assert!(
            lines[3]["result"]["structuredContent"]["qd"]["daily_dose_mg"]
                .as_f64()
                .unwrap_or(0.0)
                > 0.0
        );
        assert!(base.join("ibu_pk.svg").exists());
        assert_eq!(lines[4]["result"]["isError"], true);

        let _ = std::fs::remove_dir_all(&base);
        Ok(())
    }
}
