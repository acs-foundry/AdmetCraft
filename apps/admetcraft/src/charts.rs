use std::collections::HashMap;

use admet_calc::pharmacokinetics::{
    BOILED_EGG_WHITE_ANGLE_RAD, BOILED_EGG_WHITE_AXES, BOILED_EGG_WHITE_CENTER,
    BOILED_EGG_YOLK_ANGLE_RAD, BOILED_EGG_YOLK_AXES, BOILED_EGG_YOLK_CENTER,
};
use admet_calc::{BioavailabilityRadar, BoiledEggGeometry, MoleculeEvaluation};
use chem_core::Molecule;
use egui::{Color32, Pos2, Rect, Sense, Shape, Stroke, Ui, Vec2};

use crate::sketcher::draw_cached_molecule_2d;

/// Render the 6-axis Bioavailability Radar using shared `BioavailabilityRadar` geometry.
pub fn draw_bioavailability_radar(
    ui: &mut Ui,
    eval: &MoleculeEvaluation,
    size: f32,
    dark_mode: bool,
) {
    let (rect, _resp) = ui.allocate_exact_size(Vec2::splat(size), Sense::hover());
    let painter = ui.painter_at(rect);

    let bg = if dark_mode {
        Color32::from_rgb(30, 32, 36)
    } else {
        Color32::WHITE
    };
    painter.rect_filled(rect, 4.0, bg);

    let cx = rect.center().x;
    let cy = rect.center().y;
    let max_r = size * 0.35;

    let to_screen = |(ux, uy): (f64, f64)| -> Pos2 {
        Pos2::new(cx + (ux as f32) * max_r, cy - (uy as f32) * max_r)
    };
    let hex = |r: f64| -> Vec<Pos2> {
        BioavailabilityRadar::unit_hexagon(r)
            .into_iter()
            .map(to_screen)
            .collect()
    };

    let pink_fill = if dark_mode {
        Color32::from_rgba_unmultiplied(210, 75, 95, 65)
    } else {
        Color32::from_rgb(253, 226, 228)
    };
    painter.add(Shape::convex_polygon(
        hex(0.85),
        pink_fill,
        Stroke::new(1.0_f32, Color32::from_rgb(240, 150, 165)),
    ));
    painter.add(Shape::convex_polygon(
        hex(0.25),
        bg,
        Stroke::new(1.0_f32, Color32::from_rgb(240, 150, 165)),
    ));

    let grid_col = if dark_mode {
        Color32::from_rgb(75, 78, 86)
    } else {
        Color32::from_rgb(205, 205, 205)
    };
    for frac in [0.5, 1.0] {
        painter.add(Shape::closed_line(
            hex(frac),
            Stroke::new(0.8_f32, grid_col),
        ));
    }

    let label_col = if dark_mode {
        Color32::from_rgb(210, 214, 220)
    } else {
        Color32::from_rgb(60, 60, 65)
    };
    let full_hex = BioavailabilityRadar::unit_hexagon(1.0);
    let label_hex = BioavailabilityRadar::unit_hexagon(f64::from(max_r + 16.0) / f64::from(max_r));
    for (i, label) in BioavailabilityRadar::AXIS_LABELS.iter().enumerate() {
        let end = to_screen(full_hex[i]);
        let lpos = to_screen(label_hex[i]);
        painter.line_segment([Pos2::new(cx, cy), end], Stroke::new(0.8_f32, grid_col));
        painter.text(
            lpos,
            egui::Align2::CENTER_CENTER,
            *label,
            egui::FontId::proportional(10.0),
            label_col,
        );
    }

    let mol_pts: Vec<Pos2> = eval
        .bioavailability_radar
        .unit_polygon()
        .into_iter()
        .map(to_screen)
        .collect();
    painter.add(Shape::convex_polygon(
        mol_pts.clone(),
        Color32::from_rgba_unmultiplied(201, 42, 42, 60),
        Stroke::new(2.0_f32, Color32::from_rgb(201, 42, 42)),
    ));
    for p in mol_pts {
        painter.circle_filled(p, 3.0, Color32::from_rgb(201, 42, 42));
    }
}

/// Interactive BOILED-Egg scatter plot using shared `BoiledEggGeometry` and cached 2D molecules.
pub fn draw_boiled_egg_interactive(
    ui: &mut Ui,
    evals: &[MoleculeEvaluation],
    mol_cache: &HashMap<String, Molecule>,
    show_names: bool,
    selected_idx: Option<usize>,
    dark_mode: bool,
) -> Option<usize> {
    let avail = Vec2::new(ui.available_width().max(320.0), 420.0);
    let (full_rect, response) = ui.allocate_exact_size(avail, Sense::click());
    let painter = ui.painter_at(full_rect);

    let plot_rect = Rect::from_min_max(
        Pos2::new(full_rect.min.x + 56.0, full_rect.min.y + 18.0),
        Pos2::new(full_rect.max.x - 160.0, full_rect.max.y - 42.0),
    );
    let bg_grey = if dark_mode {
        Color32::from_rgb(65, 68, 75)
    } else {
        Color32::from_rgb(218, 218, 220)
    };
    painter.rect_filled(plot_rect, 2.0, bg_grey);
    painter.rect_stroke(
        plot_rect,
        2.0,
        Stroke::new(1.0_f32, Color32::from_rgb(120, 120, 125)),
        egui::StrokeKind::Outside,
    );

    let map_x = |tpsa: f64| -> f32 {
        (plot_rect.min.x + ((tpsa / 200.0) as f32) * plot_rect.width())
            .clamp(plot_rect.min.x, plot_rect.max.x)
    };
    let map_y = |wlogp: f64| -> f32 {
        (plot_rect.max.y - (((wlogp + 4.0) / 12.0) as f32) * plot_rect.height())
            .clamp(plot_rect.min.y, plot_rect.max.y)
    };

    let white_poly: Vec<Pos2> = BoiledEggGeometry::sample_ellipse(
        BOILED_EGG_WHITE_CENTER,
        BOILED_EGG_WHITE_AXES,
        BOILED_EGG_WHITE_ANGLE_RAD,
        72,
    )
    .into_iter()
    .map(|(x, y)| Pos2::new(map_x(x), map_y(y)))
    .collect();
    painter.add(Shape::convex_polygon(
        white_poly,
        Color32::WHITE,
        Stroke::new(1.0_f32, Color32::from_rgb(140, 140, 145)),
    ));

    let yolk_poly: Vec<Pos2> = BoiledEggGeometry::sample_ellipse(
        BOILED_EGG_YOLK_CENTER,
        BOILED_EGG_YOLK_AXES,
        BOILED_EGG_YOLK_ANGLE_RAD,
        72,
    )
    .into_iter()
    .map(|(x, y)| Pos2::new(map_x(x), map_y(y)))
    .collect();
    painter.add(Shape::convex_polygon(
        yolk_poly,
        Color32::from_rgb(245, 197, 24),
        Stroke::new(1.0_f32, Color32::from_rgb(195, 148, 0)),
    ));

    let tick_col = if dark_mode {
        Color32::from_rgb(200, 200, 205)
    } else {
        Color32::from_rgb(45, 45, 50)
    };
    for t in [0, 40, 80, 120, 160, 200] {
        let x = map_x(f64::from(t));
        painter.line_segment(
            [
                Pos2::new(x, plot_rect.max.y),
                Pos2::new(x, plot_rect.max.y + 4.0),
            ],
            Stroke::new(1.0_f32, tick_col),
        );
        painter.text(
            Pos2::new(x, plot_rect.max.y + 14.0),
            egui::Align2::CENTER_CENTER,
            t.to_string(),
            egui::FontId::proportional(10.0),
            tick_col,
        );
    }
    for lp in [-4, -2, 0, 2, 4, 6, 8] {
        let y = map_y(f64::from(lp));
        painter.line_segment(
            [
                Pos2::new(plot_rect.min.x - 4.0, y),
                Pos2::new(plot_rect.min.x, y),
            ],
            Stroke::new(1.0_f32, tick_col),
        );
        painter.text(
            Pos2::new(plot_rect.min.x - 8.0, y),
            egui::Align2::RIGHT_CENTER,
            lp.to_string(),
            egui::FontId::proportional(10.0),
            tick_col,
        );
    }
    painter.text(
        Pos2::new(plot_rect.center().x, full_rect.max.y - 10.0),
        egui::Align2::CENTER_CENTER,
        "TPSA (A^2)",
        egui::FontId::proportional(11.0),
        tick_col,
    );
    painter.text(
        Pos2::new(full_rect.min.x + 16.0, plot_rect.center().y),
        egui::Align2::CENTER_CENTER,
        "WLOGP",
        egui::FontId::proportional(11.0),
        tick_col,
    );

    let pointer_pos = response.hover_pos();
    let mut hovered_idx: Option<usize> = None;
    let mut clicked_idx: Option<usize> = None;
    let mut out_of_range = 0usize;

    for (idx, e) in evals.iter().enumerate() {
        if !e.boiled_egg.in_plot_bounds {
            out_of_range += 1;
            continue;
        }
        let p = Pos2::new(map_x(e.boiled_egg.tpsa), map_y(e.boiled_egg.wlogp));
        let col = if e.boiled_egg.pgp_substrate {
            Color32::from_rgb(28, 126, 214)
        } else {
            Color32::from_rgb(224, 49, 49)
        };
        let is_sel = selected_idx == Some(idx);
        let r = if is_sel { 6.5 } else { 5.0 };
        painter.circle_filled(p, r, col);
        painter.circle_stroke(
            p,
            r,
            Stroke::new(if is_sel { 2.0_f32 } else { 1.0_f32 }, Color32::BLACK),
        );
        if show_names {
            let label = truncate_label(&e.name, 18);
            painter.text(
                Pos2::new(p.x + 7.0, p.y - 6.0),
                egui::Align2::LEFT_CENTER,
                &label,
                egui::FontId::proportional(10.0),
                Color32::from_rgb(20, 20, 25),
            );
        }
        if let Some(hp) = pointer_pos
            && hp.distance(p) <= 9.0
        {
            hovered_idx = Some(idx);
        }
        if response.clicked()
            && let Some(cp) = response.interact_pointer_pos()
            && cp.distance(p) <= 9.0
        {
            clicked_idx = Some(idx);
        }
    }

    let lx = plot_rect.max.x + 14.0;
    let ly = plot_rect.min.y + 6.0;
    let legend_rect = Rect::from_min_size(Pos2::new(lx, ly), Vec2::new(138.0, 155.0));
    painter.rect_filled(
        legend_rect,
        4.0,
        if dark_mode {
            Color32::from_rgb(40, 42, 48)
        } else {
            Color32::from_rgb(250, 250, 252)
        },
    );
    painter.rect_stroke(
        legend_rect,
        4.0,
        Stroke::new(1.0_f32, Color32::from_rgb(170, 170, 175)),
        egui::StrokeKind::Outside,
    );
    let leg_txt = if dark_mode {
        Color32::WHITE
    } else {
        Color32::BLACK
    };
    painter.rect_filled(
        Rect::from_min_size(Pos2::new(lx + 10.0, ly + 14.0), Vec2::splat(12.0)),
        1.0,
        Color32::WHITE,
    );
    painter.text(
        Pos2::new(lx + 28.0, ly + 20.0),
        egui::Align2::LEFT_CENTER,
        "HIA (White)",
        egui::FontId::proportional(11.0),
        leg_txt,
    );
    painter.rect_filled(
        Rect::from_min_size(Pos2::new(lx + 10.0, ly + 36.0), Vec2::splat(12.0)),
        1.0,
        Color32::from_rgb(245, 197, 24),
    );
    painter.text(
        Pos2::new(lx + 28.0, ly + 42.0),
        egui::Align2::LEFT_CENTER,
        "BBB (Yolk)",
        egui::FontId::proportional(11.0),
        leg_txt,
    );
    painter.circle_filled(
        Pos2::new(lx + 16.0, ly + 66.0),
        5.0,
        Color32::from_rgb(28, 126, 214),
    );
    painter.text(
        Pos2::new(lx + 28.0, ly + 66.0),
        egui::Align2::LEFT_CENTER,
        "PGP+",
        egui::FontId::proportional(11.0),
        leg_txt,
    );
    painter.circle_filled(
        Pos2::new(lx + 16.0, ly + 88.0),
        5.0,
        Color32::from_rgb(224, 49, 49),
    );
    painter.text(
        Pos2::new(lx + 28.0, ly + 88.0),
        egui::Align2::LEFT_CENTER,
        "PGP-",
        egui::FontId::proportional(11.0),
        leg_txt,
    );
    painter.text(
        Pos2::new(lx + 10.0, ly + 124.0),
        egui::Align2::LEFT_CENTER,
        format!("Out of range: {out_of_range}"),
        egui::FontId::proportional(10.5),
        Color32::from_rgb(130, 130, 135),
    );

    if let Some(hi) = hovered_idx {
        let e = &evals[hi];
        egui::Tooltip::for_widget(&response)
            .at_pointer()
            .show(|ui| {
                ui.set_min_width(180.0);
                ui.strong(&e.name);
                ui.label(format!(
                    "WLOGP: {:.2}  |  TPSA: {:.2} A^2",
                    e.boiled_egg.wlogp, e.boiled_egg.tpsa
                ));
                ui.label(format!(
                    "GI: {}  |  BBB: {}  |  PGP (exp.): {}",
                    e.pharmacokinetics.gi_absorption.as_str(),
                    if e.pharmacokinetics.bbb_permeant {
                        "Yes"
                    } else {
                        "No"
                    },
                    if e.boiled_egg.pgp_substrate {
                        "PGP+"
                    } else {
                        "PGP-"
                    }
                ));
                if let Some(mol) = mol_cache.get(&e.canonical_smiles) {
                    let (srect, _) =
                        ui.allocate_exact_size(Vec2::new(170.0, 110.0), Sense::hover());
                    let spainter = ui.painter_at(srect);
                    draw_cached_molecule_2d(&spainter, mol, srect, dark_mode);
                }
            });
    }

    clicked_idx
}

fn truncate_label(s: &str, max_chars: usize) -> String {
    let mut chars = s.chars();
    let prefix: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{prefix}...")
    } else {
        prefix
    }
}
