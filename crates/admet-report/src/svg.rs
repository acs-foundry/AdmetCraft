use std::fmt::Write as _;

use admet_calc::pharmacokinetics::{
    BOILED_EGG_WHITE_ANGLE_RAD, BOILED_EGG_WHITE_AXES, BOILED_EGG_WHITE_CENTER,
    BOILED_EGG_YOLK_ANGLE_RAD, BOILED_EGG_YOLK_AXES, BOILED_EGG_YOLK_CENTER,
};
use admet_calc::{
    BioavailabilityRadar, BoiledEggGeometry, MoleculeEvaluation, PbpkHumanDoseReport,
};

/// Render a 6-axis Bioavailability Radar chart as standalone SVG using shared radar geometry.
pub fn render_radar_svg(eval: &MoleculeEvaluation) -> String {
    let w = 280.0;
    let h = 260.0;
    let cx = 140.0;
    let cy = 132.0;
    let max_r = 84.0;

    let mut out = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\">\n\
         <rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n"
    );

    let to_svg = |(ux, uy): (f64, f64)| (cx + ux * max_r, cy - uy * max_r);

    let outer_pts: Vec<(f64, f64)> = BioavailabilityRadar::unit_hexagon(0.85)
        .into_iter()
        .map(to_svg)
        .collect();
    let inner_pts: Vec<(f64, f64)> = BioavailabilityRadar::unit_hexagon(0.25)
        .into_iter()
        .map(to_svg)
        .collect();
    let _ = writeln!(
        out,
        "<polygon points=\"{}\" fill=\"#fde2e4\" stroke=\"#f4a6b0\" stroke-width=\"1\"/>",
        fmt_pts(&outer_pts)
    );
    let _ = writeln!(
        out,
        "<polygon points=\"{}\" fill=\"#ffffff\" stroke=\"#f4a6b0\" stroke-width=\"1\"/>",
        fmt_pts(&inner_pts)
    );

    for frac in [0.25, 0.5, 0.75, 1.0] {
        let pts: Vec<(f64, f64)> = BioavailabilityRadar::unit_hexagon(frac)
            .into_iter()
            .map(to_svg)
            .collect();
        let _ = writeln!(
            out,
            "<polygon points=\"{}\" fill=\"none\" stroke=\"#d0d0d0\" stroke-width=\"0.8\"/>",
            fmt_pts(&pts)
        );
    }

    let full_hex = BioavailabilityRadar::unit_hexagon(1.0);
    let label_hex = BioavailabilityRadar::unit_hexagon((max_r + 18.0) / max_r);
    for (i, label) in BioavailabilityRadar::AXIS_LABELS.iter().enumerate() {
        let (x2, y2) = to_svg(full_hex[i]);
        let (lx, ly) = to_svg(label_hex[i]);
        let _ = writeln!(
            out,
            "<line x1=\"{cx:.1}\" y1=\"{cy:.1}\" x2=\"{x2:.1}\" y2=\"{y2:.1}\" stroke=\"#cccccc\" stroke-width=\"0.8\"/>"
        );
        let _ = writeln!(
            out,
            "<text x=\"{lx:.1}\" y=\"{:.1}\" font-family=\"Helvetica,sans-serif\" font-size=\"10\" font-weight=\"700\" text-anchor=\"middle\" fill=\"#444444\">{label}</text>",
            ly + 4.0
        );
    }

    let mol_pts: Vec<(f64, f64)> = eval
        .bioavailability_radar
        .unit_polygon()
        .into_iter()
        .map(to_svg)
        .collect();
    let _ = writeln!(
        out,
        "<polygon points=\"{}\" fill=\"rgba(217, 72, 72, 0.25)\" stroke=\"#c92a2a\" stroke-width=\"2\"/>",
        fmt_pts(&mol_pts)
    );
    for (px, py) in &mol_pts {
        let _ = writeln!(
            out,
            "<circle cx=\"{px:.1}\" cy=\"{py:.1}\" r=\"3\" fill=\"#c92a2a\"/>"
        );
    }

    let _ = writeln!(out, "</svg>");
    out
}

/// Render a BOILED-Egg SVG scatter plot using shared `BoiledEggGeometry`.
pub fn render_boiled_egg_svg(evals: &[MoleculeEvaluation]) -> String {
    let w = 640.0;
    let h = 440.0;
    let pad_l = 60.0;
    let pad_r = 160.0;
    let pad_t = 30.0;
    let pad_b = 50.0;
    let plot_w = w - pad_l - pad_r;
    let plot_h = h - pad_t - pad_b;

    let map_x = |tpsa: f64| (pad_l + (tpsa / 200.0) * plot_w).clamp(pad_l, pad_l + plot_w);
    let map_y = |wlogp: f64| {
        (pad_t + plot_h - ((wlogp + 4.0) / 12.0) * plot_h).clamp(pad_t, pad_t + plot_h)
    };

    let mut out = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\">\n\
         <rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n\
         <rect x=\"{pad_l}\" y=\"{pad_t}\" width=\"{plot_w}\" height=\"{plot_h}\" fill=\"#d9d9d9\" stroke=\"#555555\" stroke-width=\"1\"/>\n"
    );

    let white_pts: Vec<(f64, f64)> = BoiledEggGeometry::sample_ellipse(
        BOILED_EGG_WHITE_CENTER,
        BOILED_EGG_WHITE_AXES,
        BOILED_EGG_WHITE_ANGLE_RAD,
        72,
    )
    .into_iter()
    .map(|(x, y)| (map_x(x), map_y(y)))
    .collect();
    let _ = writeln!(
        out,
        "<polygon points=\"{}\" fill=\"#ffffff\" stroke=\"#888888\" stroke-width=\"1\"/>",
        fmt_pts(&white_pts)
    );

    let yolk_pts: Vec<(f64, f64)> = BoiledEggGeometry::sample_ellipse(
        BOILED_EGG_YOLK_CENTER,
        BOILED_EGG_YOLK_AXES,
        BOILED_EGG_YOLK_ANGLE_RAD,
        72,
    )
    .into_iter()
    .map(|(x, y)| (map_x(x), map_y(y)))
    .collect();
    let _ = writeln!(
        out,
        "<polygon points=\"{}\" fill=\"#f5c518\" stroke=\"#c99a00\" stroke-width=\"1\"/>",
        fmt_pts(&yolk_pts)
    );

    for t in [0, 40, 80, 120, 160, 200] {
        let x = map_x(f64::from(t));
        let y0 = pad_t + plot_h;
        let _ = writeln!(
            out,
            "<line x1=\"{x:.1}\" y1=\"{y0:.1}\" x2=\"{x:.1}\" y2=\"{:.1}\" stroke=\"#444444\"/>",
            y0 + 5.0
        );
        let _ = writeln!(
            out,
            "<text x=\"{x:.1}\" y=\"{:.1}\" font-family=\"Helvetica,sans-serif\" font-size=\"10\" text-anchor=\"middle\" fill=\"#333333\">{t}</text>",
            y0 + 18.0
        );
    }
    for lp in [-4, -2, 0, 2, 4, 6, 8] {
        let y = map_y(f64::from(lp));
        let _ = writeln!(
            out,
            "<line x1=\"{:.1}\" y1=\"{y:.1}\" x2=\"{pad_l:.1}\" y2=\"{y:.1}\" stroke=\"#444444\"/>",
            pad_l - 5.0
        );
        let _ = writeln!(
            out,
            "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"Helvetica,sans-serif\" font-size=\"10\" text-anchor=\"end\" fill=\"#333333\">{lp}</text>",
            pad_l - 8.0,
            y + 3.0
        );
    }

    let _ = writeln!(
        out,
        "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"Helvetica,sans-serif\" font-size=\"11\" font-weight=\"700\" text-anchor=\"middle\" fill=\"#222222\">TPSA (A^2)</text>",
        pad_l + plot_w * 0.5,
        h - 12.0
    );
    let _ = writeln!(
        out,
        "<text x=\"18\" y=\"{:.1}\" transform=\"rotate(-90 18 {:.1})\" font-family=\"Helvetica,sans-serif\" font-size=\"11\" font-weight=\"700\" text-anchor=\"middle\" fill=\"#222222\">WLOGP</text>",
        pad_t + plot_h * 0.5,
        pad_t + plot_h * 0.5
    );

    let mut out_of_range = 0usize;
    for e in evals {
        if !e.boiled_egg.in_plot_bounds {
            out_of_range += 1;
            continue;
        }
        let px = map_x(e.boiled_egg.tpsa);
        let py = map_y(e.boiled_egg.wlogp);
        let color = if e.boiled_egg.pgp_substrate {
            "#1c7ed6"
        } else {
            "#e03131"
        };
        let _ = writeln!(
            out,
            "<circle cx=\"{px:.1}\" cy=\"{py:.1}\" r=\"5\" fill=\"{color}\" stroke=\"#111111\" stroke-width=\"0.9\"/>"
        );
        let safe_name = escape_xml(&truncate_label(&e.name, 36));
        let _ = writeln!(
            out,
            "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"Helvetica,sans-serif\" font-size=\"9\" fill=\"#111111\">{safe_name}</text>",
            px + 7.0,
            py + 3.0
        );
    }

    let lx = pad_l + plot_w + 15.0;
    let _ = writeln!(
        out,
        "<rect x=\"{lx}\" y=\"40\" width=\"135\" height=\"150\" fill=\"#fbfbfb\" stroke=\"#cccccc\"/>\n\
         <rect x=\"{:.1}\" y=\"54\" width=\"12\" height=\"12\" fill=\"#ffffff\" stroke=\"#888888\"/>\n\
         <text x=\"{:.1}\" y=\"64\" font-family=\"Helvetica,sans-serif\" font-size=\"10\" fill=\"#222\">HIA (White)</text>\n\
         <rect x=\"{:.1}\" y=\"74\" width=\"12\" height=\"12\" fill=\"#f5c518\" stroke=\"#c99a00\"/>\n\
         <text x=\"{:.1}\" y=\"84\" font-family=\"Helvetica,sans-serif\" font-size=\"10\" fill=\"#222\">BBB (Yolk)</text>\n\
         <circle cx=\"{:.1}\" cy=\"104\" r=\"4.5\" fill=\"#1c7ed6\" stroke=\"#111\"/>\n\
         <text x=\"{:.1}\" y=\"108\" font-family=\"Helvetica,sans-serif\" font-size=\"10\" fill=\"#222\">PGP+</text>\n\
         <circle cx=\"{:.1}\" cy=\"124\" r=\"4.5\" fill=\"#e03131\" stroke=\"#111\"/>\n\
         <text x=\"{:.1}\" y=\"128\" font-family=\"Helvetica,sans-serif\" font-size=\"10\" fill=\"#222\">PGP-</text>\n\
         <text x=\"{:.1}\" y=\"158\" font-family=\"Helvetica,sans-serif\" font-size=\"9\" fill=\"#555\">Out of range: {out_of_range}</text>",
        lx + 10.0,
        lx + 28.0,
        lx + 10.0,
        lx + 28.0,
        lx + 16.0,
        lx + 28.0,
        lx + 16.0,
        lx + 28.0,
        lx + 10.0
    );

    let _ = writeln!(out, "</svg>");
    out
}

/// Renders a standalone vector SVG of the steady-state 48 h unbound plasma concentration–time
/// profile (`C_u(t)`) with 90% conformal uncertainty ribbon, `IC50`/`IC90` target lines, and
/// human dosing summary box.
#[must_use]
pub fn render_pk_profile_svg(report: &PbpkHumanDoseReport) -> String {
    let w = 680.0_f64;
    let h = 360.0_f64;
    let pad_l = 66.0_f64;
    let pad_r = 180.0_f64;
    let pad_t = 44.0_f64;
    let pad_b = 50.0_f64;
    let plot_w = w - pad_l - pad_r;
    let plot_h = h - pad_t - pad_b;

    let max_t = report
        .steady_state_curve
        .last()
        .map_or(48.0, |p| p.time_h.max(24.0));
    let ic50 = report.target_potency_nm.max(0.1);
    let ic90 = ic50 * 9.0;
    let max_curve = report
        .steady_state_curve
        .iter()
        .map(|p| p.cu_p95_nm.max(p.cu_median_nm))
        .fold(ic50 * 2.0, f64::max);
    let y_max = (max_curve.max(ic50 * 2.5) * 1.15).max(1.0);

    let map_x = |t: f64| pad_l + (t / max_t).clamp(0.0, 1.0) * plot_w;
    let map_y = |c: f64| pad_t + plot_h - (c / y_max).clamp(0.0, 1.0) * plot_h;

    let safe_title = escape_xml(&truncate_label(&report.molecule_name, 38));
    let mut out = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\">\n\
         <rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n\
         <text x=\"{pad_l}\" y=\"24\" font-family=\"Helvetica,sans-serif\" font-size=\"13\" font-weight=\"700\" fill=\"#111827\">Human Steady-State Unbound Plasma PK: {safe_title}</text>\n\
         <rect x=\"{pad_l}\" y=\"{pad_t}\" width=\"{plot_w}\" height=\"{plot_h}\" fill=\"#f8fafc\" stroke=\"#94a3b8\" stroke-width=\"1\"/>\n"
    );

    // Horizontal grid lines & Y axis labels
    for i in 0..=4 {
        let frac = (i as f64) / 4.0;
        let val = y_max * frac;
        let yy = map_y(val);
        let _ = writeln!(
            out,
            "<line x1=\"{pad_l:.1}\" y1=\"{yy:.1}\" x2=\"{:.1}\" y2=\"{yy:.1}\" stroke=\"#e2e8f0\" stroke-width=\"0.8\"/>\n\
             <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"Helvetica,sans-serif\" font-size=\"9.5\" text-anchor=\"end\" fill=\"#475569\">{val:.0}</text>",
            pad_l + plot_w,
            pad_l - 6.0,
            yy + 3.5
        );
    }

    // Vertical grid lines (every 12 h)
    let mut t_tick = 0.0;
    while t_tick <= max_t + 0.1 {
        let xx = map_x(t_tick);
        let _ = writeln!(
            out,
            "<line x1=\"{xx:.1}\" y1=\"{pad_t:.1}\" x2=\"{xx:.1}\" y2=\"{:.1}\" stroke=\"#e2e8f0\" stroke-width=\"0.8\"/>\n\
             <text x=\"{xx:.1}\" y=\"{:.1}\" font-family=\"Helvetica,sans-serif\" font-size=\"9.5\" text-anchor=\"middle\" fill=\"#475569\">{t_tick:.0}h</text>",
            pad_t + plot_h,
            pad_t + plot_h + 16.0
        );
        t_tick += 12.0;
    }

    // 90% conformal uncertainty ribbon (p5..p95)
    if report.steady_state_curve.len() >= 2 {
        let mut poly_pts: Vec<(f64, f64)> = report
            .steady_state_curve
            .iter()
            .map(|p| (map_x(p.time_h), map_y(p.cu_p95_nm)))
            .collect();
        for p in report.steady_state_curve.iter().rev() {
            poly_pts.push((map_x(p.time_h), map_y(p.cu_p5_nm)));
        }
        let _ = writeln!(
            out,
            "<polygon points=\"{}\" fill=\"rgba(37, 99, 235, 0.16)\" stroke=\"none\"/>",
            fmt_pts(&poly_pts)
        );
        let med_pts: Vec<(f64, f64)> = report
            .steady_state_curve
            .iter()
            .map(|p| (map_x(p.time_h), map_y(p.cu_median_nm)))
            .collect();
        let _ = writeln!(
            out,
            "<polyline points=\"{}\" fill=\"none\" stroke=\"#1d4ed8\" stroke-width=\"2.2\"/>",
            fmt_pts(&med_pts)
        );
    }

    // IC50 & IC90 target reference lines
    let y_ic50 = map_y(ic50);
    let _ = writeln!(
        out,
        "<line x1=\"{pad_l:.1}\" y1=\"{y_ic50:.1}\" x2=\"{:.1}\" y2=\"{y_ic50:.1}\" stroke=\"#16a34a\" stroke-width=\"1.4\" stroke-dasharray=\"5,4\"/>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"Helvetica,sans-serif\" font-size=\"9\" font-weight=\"700\" fill=\"#15803d\">IC50 ({ic50:.1} nM)</text>",
        pad_l + plot_w,
        pad_l + 6.0,
        (y_ic50 - 4.0).max(pad_t + 10.0)
    );
    if ic90 <= y_max * 0.98 {
        let y_ic90 = map_y(ic90);
        let _ = writeln!(
            out,
            "<line x1=\"{pad_l:.1}\" y1=\"{y_ic90:.1}\" x2=\"{:.1}\" y2=\"{y_ic90:.1}\" stroke=\"#ea580c\" stroke-width=\"1.2\" stroke-dasharray=\"3,3\"/>\n\
             <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"Helvetica,sans-serif\" font-size=\"9\" font-weight=\"700\" fill=\"#c2410c\">IC90 ({ic90:.0} nM)</text>",
            pad_l + plot_w,
            pad_l + 6.0,
            (y_ic90 - 4.0).max(pad_t + 10.0)
        );
    }

    // Axis titles
    let _ = writeln!(
        out,
        "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"Helvetica,sans-serif\" font-size=\"10.5\" font-weight=\"700\" text-anchor=\"middle\" fill=\"#1e293b\">Steady-State Time (hours)</text>\n\
         <text x=\"18\" y=\"{:.1}\" transform=\"rotate(-90 18 {:.1})\" font-family=\"Helvetica,sans-serif\" font-size=\"10.5\" font-weight=\"700\" text-anchor=\"middle\" fill=\"#1e293b\">Unbound Plasma C_u (nM)</text>",
        pad_l + plot_w * 0.5,
        h - 12.0,
        pad_t + plot_h * 0.5,
        pad_t + plot_h * 0.5
    );

    // Right-hand summary card
    let sx = pad_l + plot_w + 14.0;
    let safe_bcs = escape_xml(report.bcs_class.short_label());
    let _ = writeln!(
        out,
        "<rect x=\"{sx:.1}\" y=\"{pad_t:.1}\" width=\"156\" height=\"240\" rx=\"6\" fill=\"#f8fafc\" stroke=\"#cbd5e1\"/>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"Helvetica,sans-serif\" font-size=\"10.5\" font-weight=\"700\" fill=\"#0f172a\">Human PK Summary</text>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"Helvetica,sans-serif\" font-size=\"9.5\" fill=\"#334155\">Route: {} | {safe_bcs}</text>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"Helvetica,sans-serif\" font-size=\"9.5\" fill=\"#334155\">F_oral: {:.1}% | f_u,p: {:.3}</text>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"Helvetica,sans-serif\" font-size=\"9.5\" fill=\"#334155\">CL_tot: {:.2} mL/min/kg</text>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"Helvetica,sans-serif\" font-size=\"9.5\" fill=\"#334155\">V_d,ss: {:.2} L/kg</text>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"Helvetica,sans-serif\" font-size=\"9.5\" fill=\"#334155\">t1/2: {:.1} h</text>\n\
         <line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#cbd5e1\"/>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"Helvetica,sans-serif\" font-size=\"9.5\" font-weight=\"700\" fill=\"#1d4ed8\">QD: {:.1} mg/d</text>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"Helvetica,sans-serif\" font-size=\"8.5\" fill=\"#475569\">90% CI: {:.0}-{:.0} mg</text>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"Helvetica,sans-serif\" font-size=\"9.5\" font-weight=\"700\" fill=\"#0f766e\">BID: {:.1} mg q12h</text>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"Helvetica,sans-serif\" font-size=\"8.5\" fill=\"#475569\">90% CI: {:.0}-{:.0} mg/d</text>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"Helvetica,sans-serif\" font-size=\"9\" fill=\"#334155\">hERG Margin: {:.0}x</text>\n\
         <rect x=\"{:.1}\" y=\"{:.1}\" width=\"12\" height=\"8\" fill=\"rgba(37, 99, 235, 0.25)\"/>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"Helvetica,sans-serif\" font-size=\"8.5\" fill=\"#475569\">90% Conformal Band</text>",
        sx + 10.0,
        pad_t + 20.0,
        sx + 10.0,
        pad_t + 40.0,
        report.route.as_str(),
        sx + 10.0,
        pad_t + 58.0,
        report.oral_bioavailability_f * 100.0,
        report.fup,
        sx + 10.0,
        pad_t + 76.0,
        report.cl_total_ml_min_kg,
        sx + 10.0,
        pad_t + 94.0,
        report.vdss_l_kg,
        sx + 10.0,
        pad_t + 112.0,
        report.t_half_ivive_h,
        sx + 8.0,
        pad_t + 124.0,
        sx + 148.0,
        pad_t + 124.0,
        sx + 10.0,
        pad_t + 142.0,
        report.qd.daily_dose_mg,
        sx + 10.0,
        pad_t + 156.0,
        report.qd.daily_dose_ci90_mg[0],
        report.qd.daily_dose_ci90_mg[1],
        sx + 10.0,
        pad_t + 176.0,
        report.bid.dose_mg_per_admin,
        sx + 10.0,
        pad_t + 190.0,
        report.bid.daily_dose_ci90_mg[0],
        report.bid.daily_dose_ci90_mg[1],
        sx + 10.0,
        pad_t + 210.0,
        report.qd.herg_safety_margin,
        sx + 10.0,
        pad_t + 222.0,
        sx + 26.0,
        pad_t + 230.0
    );

    let _ = writeln!(out, "</svg>");
    out
}

fn fmt_pts(pts: &[(f64, f64)]) -> String {
    pts.iter()
        .map(|(x, y)| format!("{x:.1},{y:.1}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn truncate_label(s: &str, max_chars: usize) -> String {
    let cnt = s.chars().count();
    if cnt <= max_chars {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max_chars.saturating_sub(3)).collect();
        out.push_str("...");
        out
    }
}

fn escape_xml(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        // Strip ASCII control characters and XML 1.0 Unicode noncharacters (U+FDD0..=U+FDEF, U+FFFE, U+FFFF)
        if c.is_control() || matches!(c, '\u{FDD0}'..='\u{FDEF}' | '\u{FFFE}' | '\u{FFFF}') {
            out.push(' ');
            continue;
        }
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}
