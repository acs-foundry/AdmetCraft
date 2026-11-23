//! Layer 2 (`admet-report`): Vector charts (Bioavailability Radar, BOILED-Egg) and
//! CSV, JSON, and publication-ready PDF report generation.

#![forbid(unsafe_code)]

pub mod csv_json;
pub mod pdf;
pub mod svg;

use std::io::Write;
use std::path::Path;

use admet_calc::MoleculeEvaluation;
use thiserror::Error;

pub use csv_json::{export_csv_string, export_json_string};
pub use pdf::export_pdf_bytes;
pub use svg::{render_boiled_egg_svg, render_pk_profile_svg, render_radar_svg};

/// Errors produced by `admet-report`.
#[derive(Debug, Error)]
pub enum ReportError {
    #[error("cannot generate report for empty molecule list")]
    EmptyReport,
    #[error("CSV serialization error: {0}")]
    Csv(#[from] csv::Error),
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("formatting error: {0}")]
    Format(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Supported export formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    Csv,
    Json,
    Pdf,
    SvgBoiledEgg,
}

/// Write an evaluation report directly into an open `Write` stream (eliminating path TOCTOU).
pub fn write_report_to_writer<W: Write>(
    evals: &[MoleculeEvaluation],
    writer: &mut W,
    format: ExportFormat,
) -> Result<(), ReportError> {
    if evals.is_empty() {
        return Err(ReportError::EmptyReport);
    }
    match format {
        ExportFormat::Csv => {
            let text = export_csv_string(evals)?;
            writer.write_all(text.as_bytes())?;
        }
        ExportFormat::Json => {
            let text = export_json_string(evals)?;
            writer.write_all(text.as_bytes())?;
        }
        ExportFormat::Pdf => {
            let bytes = export_pdf_bytes(evals)?;
            writer.write_all(&bytes)?;
        }
        ExportFormat::SvgBoiledEgg => {
            let text = render_boiled_egg_svg(evals);
            writer.write_all(text.as_bytes())?;
        }
    }
    writer.flush()?;
    Ok(())
}

/// Write an evaluation report to `path` in the requested `format`.
pub fn write_report_file(
    evals: &[MoleculeEvaluation],
    path: &Path,
    format: ExportFormat,
) -> Result<(), ReportError> {
    if evals.is_empty() {
        return Err(ReportError::EmptyReport);
    }
    let mut f = std::fs::File::create(path)?;
    write_report_to_writer(evals, &mut f, format)
}

#[cfg(test)]
mod tests {
    use super::*;
    use admet_calc::evaluate_smiles;
    use chem_core::parse_csv_batch;

    fn inflate_pdf_streams(pdf_bytes: &[u8]) -> String {
        use flate2::read::ZlibDecoder;
        use std::io::Read;
        let mut combined = String::new();
        let needle_start = b"stream\n";
        let needle_end = b"\nendstream";
        let mut pos = 0usize;
        while let Some(rel) = pdf_bytes[pos..]
            .windows(needle_start.len())
            .position(|w| w == needle_start)
        {
            let s_start = pos + rel + needle_start.len();
            let Some(e_rel) = pdf_bytes[s_start..]
                .windows(needle_end.len())
                .position(|w| w == needle_end)
            else {
                break;
            };
            let chunk = &pdf_bytes[s_start..s_start + e_rel];
            let mut dec = ZlibDecoder::new(chunk);
            let mut out = String::new();
            if dec.read_to_string(&mut out).is_ok() {
                combined.push_str(&out);
            } else {
                combined.push_str(&String::from_utf8_lossy(chunk));
            }
            pos = s_start + e_rel + needle_end.len();
        }
        combined
    }

    #[test]
    fn exports_csv_json_svg_and_valid_pdf() -> Result<(), Box<dyn std::error::Error>> {
        let evals = vec![
            evaluate_smiles("CC(=O)Oc1ccccc1C(=O)O", Some("Aspirina, ácido"))?,
            evaluate_smiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C", Some("Cafeína"))?,
        ];
        let csv_str = export_csv_string(&evals)?;
        assert!(csv_str.contains("Aspirina, ácido"));
        assert!(csv_str.contains("Consensus Log P"));
        assert!(csv_str.contains("QED (Bickerton 2012)"));
        assert!(csv_str.contains("CNS MPO (Pfizer)"));

        // Verify exported CSV roundtrips cleanly back through chem_core::parse_csv_batch!
        let reimported = parse_csv_batch(&csv_str)?;
        assert_eq!(reimported.len(), 2);
        let first = reimported
            .into_iter()
            .next()
            .ok_or(ReportError::EmptyReport)??;
        assert_eq!(first.name.as_deref(), Some("Aspirina, ácido"));

        let json_str = export_json_string(&evals)?;
        assert!(json_str.contains("bioavailability_radar"));

        let radar = render_radar_svg(&evals[0]);
        assert!(radar.contains("<svg") && radar.contains("LIPO"));

        let egg = render_boiled_egg_svg(&evals);
        assert!(egg.contains("<svg") && egg.contains("BBB (Yolk)"));

        let pdf_bytes = export_pdf_bytes(&evals)?;
        assert!(pdf_bytes.starts_with(b"%PDF-1.4"));
        assert!(pdf_bytes.ends_with(b"%%EOF\n"));
        let pdf_raw = String::from_utf8_lossy(&pdf_bytes);
        assert!(pdf_raw.contains("/Encoding /WinAnsiEncoding"));
        assert!(pdf_raw.contains("/Filter /FlateDecode"));
        let inflated = inflate_pdf_streams(&pdf_bytes);
        // Cafeína uses octal \355 for í instead of '?'
        assert!(inflated.contains("Cafe\\355na"));
        assert!(inflated.contains("QED / CNS MPO"));
        Ok(())
    }

    #[test]
    fn pdf_paginates_more_than_18_molecules() -> Result<(), Box<dyn std::error::Error>> {
        let mut evals = Vec::new();
        for i in 1..=30 {
            evals.push(evaluate_smiles("CCO", Some(&format!("Mol_{i}")))?);
        }
        let pdf_bytes = export_pdf_bytes(&evals)?;
        let inflated = inflate_pdf_streams(&pdf_bytes);
        // Continuation summary page must contain Mol_25..Mol_30 (PDF escapes parens)!
        assert!(inflated.contains("Summary Table \\(continued: 25..30 of 30\\)"));
        assert!(inflated.contains("Mol_30"));
        Ok(())
    }

    #[test]
    fn csv_sanitizes_formula_injection_and_svg_strips_control_chars()
    -> Result<(), Box<dyn std::error::Error>> {
        let dangerous_name = "=CMD|' /C calc'!A0\x00\x1b<script>&\"\u{FFFE}\u{FFFF}";
        let eval = evaluate_smiles("CCO", Some(dangerous_name))?;
        let csv_out = export_csv_string(std::slice::from_ref(&eval))?;
        assert!(
            csv_out.contains("'=CMD|"),
            "expected formula-injection prefix in CSV, got: {csv_out}"
        );

        let svg = render_boiled_egg_svg(&[eval]);
        assert!(
            !svg.chars()
                .any(|c| c.is_control() && c != '\n' && c != '\r')
        );
        assert!(!svg.contains('\u{FFFE}') && !svg.contains('\u{FFFF}'));
        assert!(svg.contains("&lt;script&gt;&amp;&quot;"));

        let dummy_reg = admet_calc::RegimenDoseEstimate {
            regimen_label: "QD (q24h)".to_string(),
            tau_h: 24.0,
            dose_mg_per_admin: 120.0,
            daily_dose_mg: 120.0,
            daily_dose_ci90_mg: [60.0, 240.0],
            cmax_u_nm: 45.0,
            cmin_u_nm: 12.0,
            cavg_u_nm: 25.0,
            auc_0_24_u_nm_h: 600.0,
            tmax_h: 2.0,
            accumulation_ratio: 1.4,
            cbrain_avg_u_nm: 5.0,
            dose_number_do: 0.4,
            solubility_limited: false,
            herg_safety_margin: 150.0,
            feasibility: "Feasible".to_string(),
        };
        let pk_rep = admet_calc::PbpkHumanDoseReport {
            molecule_name: "<script>Lead</script>".to_string(),
            canonical_smiles: "CCO".to_string(),
            molecular_weight: 350.0,
            route: admet_calc::PkRoute::Oral,
            bcs_class: admet_calc::BcsClass::ClassI,
            fup: 0.1,
            fu_inc: 0.6,
            blood_to_plasma_rb: 0.85,
            cl_int_u_ul_min_mg: 25.0,
            cl_hepatic_ml_min_kg: 3.5,
            cl_renal_ml_min_kg: 0.1,
            cl_total_ml_min_kg: 3.6,
            hepatic_extraction_eh: 0.18,
            vdss_l_kg: 2.0,
            t_half_ivive_h: 6.4,
            t_half_obach_h: Some(7.0),
            t_half_consensus_h: 6.7,
            peff_1e4_cm_s: 3.0,
            fa: 0.92,
            fg: 0.85,
            fh: 0.82,
            oral_bioavailability_f: 0.64,
            ka_per_h: 1.2,
            kpuu_brain: 0.2,
            target_potency_nm: 10.0,
            target_fold: 1.0,
            match_trough: true,
            qd: dummy_reg.clone(),
            bid: dummy_reg,
            recommended_regimen: "120 mg QD".to_string(),
            steady_state_curve: vec![
                admet_calc::PkTimePoint {
                    time_h: 0.0,
                    cu_median_nm: 12.0,
                    cu_p5_nm: 6.0,
                    cu_p95_nm: 24.0,
                },
                admet_calc::PkTimePoint {
                    time_h: 24.0,
                    cu_median_nm: 45.0,
                    cu_p5_nm: 22.0,
                    cu_p95_nm: 80.0,
                },
            ],
            warnings: Vec::new(),
            qsar_pack_loaded: true,
            estimated_inputs: Vec::new(),
        };
        let pk_svg = render_pk_profile_svg(&pk_rep);
        assert!(
            pk_svg.contains("<svg") && pk_svg.contains("IC50") && pk_svg.contains("&lt;script&gt;")
        );
        Ok(())
    }
}
