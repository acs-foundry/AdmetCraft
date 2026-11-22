#![forbid(unsafe_code)]

mod charts;
mod sketcher;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use admet_calc::{
    ApplicabilityDomainStatus, MetabolismRisk, MoleculeEvaluation, PkRoute, PkSimulationConfig,
    QsarTaskKind, TppPreset, TppStatus,
};
use admet_engine::{
    AdmetSession, Command, CommandRegistry, FilterCriteria, MAX_READ_BYTES, PathSandbox,
};
use admet_qsar::simulate_human_pk_dose;
use admet_report::{ExportFormat, write_report_to_writer};
use chem_core::{
    Molecule, ensure_2d_coords, parse_csv_batch, parse_sdf, parse_smiles, parse_smiles_batch,
};
use eframe::egui;
use egui::{Color32, RichText, Sense, Vec2};

use charts::{draw_bioavailability_radar, draw_boiled_egg_interactive};
use sketcher::{MoleculeSketcher, draw_cached_molecule_2d, draw_cached_molecule_2d_with_som};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MainTab {
    Cards,
    Qsar,
    BoiledEgg,
    Table,
}

struct AdmetCraftApp {
    session: AdmetSession,
    mol_cache: HashMap<String, Molecule>,
    sketcher: MoleculeSketcher,
    active_tab: MainTab,
    dark_mode: bool,
    show_egg_names: bool,
    show_som_overlay: bool,
    pk_potency_nm: f64,
    pk_route: PkRoute,
    palette_open: bool,
    palette_query: String,
    import_path: String,
    export_dir: String,
    status_msg: String,
    async_status: Arc<Mutex<Option<String>>>,
}

impl Default for AdmetCraftApp {
    fn default() -> Self {
        Self::from_startup_path(None)
    }
}

impl AdmetCraftApp {
    fn from_startup_path(startup_file: Option<PathBuf>) -> Self {
        let mut session = AdmetSession::new();
        let status_msg = if startup_file.is_some() {
            "Loading handoff batch...".to_string()
        } else {
            match CommandRegistry::execute(&mut session, Command::LoadExampleBatch) {
                Ok(out) => out.message,
                Err(e) => format!("Init warning: {e}"),
            }
        };
        let export_dir = startup_file
            .as_ref()
            .and_then(|p| p.parent())
            .filter(|p| !p.as_os_str().is_empty())
            .map(|p| p.display().to_string())
            .or_else(|| {
                std::env::current_dir()
                    .ok()
                    .map(|p| p.display().to_string())
            })
            .unwrap_or_else(|| ".".to_string());
        let mut sketcher = MoleculeSketcher::default();
        if let Some(first) = session.evaluations.first() {
            let _ = sketcher.load_smiles(&first.canonical_smiles);
        }
        let mut app = Self {
            session,
            mol_cache: HashMap::new(),
            sketcher,
            active_tab: MainTab::Cards,
            dark_mode: false,
            show_egg_names: true,
            show_som_overlay: true,
            pk_potency_nm: 10.0,
            pk_route: PkRoute::Oral,
            palette_open: false,
            palette_query: String::new(),
            import_path: String::new(),
            export_dir,
            status_msg,
            async_status: Arc::new(Mutex::new(None)),
        };
        if let Some(path) = startup_file {
            app.import_path = path.display().to_string();
            app.import_file_path(&path, true);
        } else {
            app.rebuild_mol_cache();
        }
        app
    }

    fn rebuild_mol_cache(&mut self) {
        self.mol_cache.clear();
        for ev in &self.session.evaluations {
            if self.mol_cache.contains_key(&ev.canonical_smiles) {
                continue;
            }
            if let Ok(mut mol) = parse_smiles(&ev.canonical_smiles) {
                ensure_2d_coords(&mut mol);
                self.mol_cache.insert(ev.canonical_smiles.clone(), mol);
            }
        }
    }

    fn sync_sketcher_to_selection(&mut self) {
        let selected_eval = self
            .session
            .selected_index
            .and_then(|idx| self.session.evaluations.get(idx))
            .or_else(|| self.session.evaluations.first());
        if let Some(ev) = selected_eval {
            let _ = self.sketcher.load_smiles(&ev.canonical_smiles);
        } else {
            self.sketcher.clear();
        }
    }

    fn run_command(&mut self, cmd: Command) {
        let should_sync_sketcher = matches!(
            cmd,
            Command::Undo
                | Command::Redo
                | Command::FilterBatch { .. }
                | Command::ClearSession
                | Command::LoadExampleBatch
                | Command::ImportSdf { .. }
                | Command::ImportCsv { .. }
                | Command::SelectMolecule { .. }
        );
        match CommandRegistry::execute(&mut self.session, cmd) {
            Ok(out) => {
                self.status_msg = out.message;
                self.rebuild_mol_cache();
                if should_sync_sketcher || self.session.evaluations.is_empty() {
                    self.sync_sketcher_to_selection();
                }
            }
            Err(e) => self.status_msg = format!("Error: {e}"),
        }
    }

    /// Export asynchronously on a background thread using `PathSandbox::open_write` so the UI
    /// thread never freezes on large PDFs and symlink/traversal escapes are blocked.
    fn export(&mut self, ctx: &egui::Context, format: ExportFormat, filename: &str) {
        if self.session.evaluations.is_empty() {
            self.status_msg = "No molecules to export".to_string();
            return;
        }
        let export_root = PathBuf::from(&self.export_dir);
        let sandbox = match PathSandbox::new(None, Some(&export_root)) {
            Ok(sb) => sb,
            Err(e) => {
                self.status_msg = format!("Export directory error: {e}");
                return;
            }
        };
        let target_name = PathBuf::from(filename);
        let evals = self.session.evaluations.clone();
        let status_slot = Arc::clone(&self.async_status);
        let repaint_ctx = ctx.clone();
        self.status_msg = format!("Exporting {filename} in background...");

        std::thread::spawn(move || {
            let msg = match sandbox.open_write(&target_name) {
                Ok((resolved, mut file)) => match write_report_to_writer(&evals, &mut file, format)
                {
                    Ok(()) => format!("Exported {}", resolved.display()),
                    Err(e) => format!("Export error: {e}"),
                },
                Err(e) => format!("Sandbox export error: {e}"),
            };
            if let Ok(mut guard) = status_slot.lock() {
                *guard = Some(msg);
            }
            repaint_ctx.request_repaint();
        });
    }

    /// Open a native Linux file dialog (`zenity` or `kdialog`) without adding external crate deps.
    fn browse_import_file(&mut self) {
        let picked = std::process::Command::new("zenity")
            .args([
                "--file-selection",
                "--title=Import Molecules (.smi, .csv, .sdf)",
                "--file-filter=Chemical files | *.smi *.smiles *.csv *.sdf *.mol *.txt",
            ])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .or_else(|| {
                std::process::Command::new("kdialog")
                    .args([
                        "--getopenfilename",
                        &self.export_dir,
                        "*.smi *.smiles *.csv *.sdf *.mol | Chemical files",
                    ])
                    .output()
                    .ok()
                    .filter(|o| o.status.success())
            });

        let Some(output) = picked else {
            self.status_msg =
                "File picker cancelled (or install zenity/kdialog, or drag-and-drop a file)"
                    .to_string();
            return;
        };
        let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if raw.is_empty() {
            return;
        }
        self.import_path = raw.clone();
        let path = PathBuf::from(raw);
        self.import_file_path(&path, true);
    }

    /// Import a `.smi`, `.csv`, or `.sdf` file inside `export_dir` (or the file's parent for
    /// explicit drag-and-drop) using `PathSandbox::read_to_string_limited` (rejects symlinks/FIFOs).
    fn import_file_path(&mut self, path: &Path, from_drop: bool) {
        let root = if from_drop {
            path.parent().unwrap_or_else(|| Path::new("."))
        } else {
            Path::new(&self.export_dir)
        };
        let sandbox = match PathSandbox::new(Some(root), None) {
            Ok(sb) => sb,
            Err(e) => {
                self.status_msg = format!("Import root error: {e}");
                return;
            }
        };
        let rel_or_abs = if from_drop {
            path.file_name().map(Path::new).unwrap_or(path)
        } else {
            path
        };
        let (resolved, content) = match sandbox.read_to_string_limited(rel_or_abs, MAX_READ_BYTES) {
            Ok(res) => res,
            Err(e) => {
                self.status_msg = format!("Sandbox import error: {e}");
                return;
            }
        };
        let ext = resolved
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let cmd = if ext == "sdf" || ext == "mol" {
            Command::ImportSdf { content }
        } else if ext == "csv" {
            Command::ImportCsv { content }
        } else {
            Command::EvaluateBatchText { input: content }
        };
        self.run_command(cmd);
    }

    fn import_dropped_files(&mut self, paths: &[PathBuf]) {
        if paths.is_empty() {
            return;
        }
        if paths.len() == 1 {
            self.import_file_path(&paths[0], true);
            return;
        }

        let mut combined_lines = Vec::new();
        let mut loaded_files = 0usize;
        for path in paths {
            let root = path.parent().unwrap_or_else(|| Path::new("."));
            let sandbox = match PathSandbox::new(Some(root), None) {
                Ok(sb) => sb,
                Err(e) => {
                    self.status_msg = format!("Import root error: {e}");
                    return;
                }
            };
            let rel = path.file_name().map(Path::new).unwrap_or(path.as_path());
            let (resolved, content) = match sandbox.read_to_string_limited(rel, MAX_READ_BYTES) {
                Ok(res) => res,
                Err(e) => {
                    self.status_msg = format!("Sandbox import error ({}): {e}", path.display());
                    return;
                }
            };
            let ext = resolved
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            let parsed_mols: Vec<Molecule> = if ext == "sdf" || ext == "mol" {
                parse_sdf(&content)
                    .into_iter()
                    .filter_map(Result::ok)
                    .collect()
            } else if ext == "csv" {
                match parse_csv_batch(&content) {
                    Ok(items) => items.into_iter().filter_map(Result::ok).collect(),
                    Err(e) => {
                        self.status_msg = format!("CSV parse error ({}): {e}", resolved.display());
                        return;
                    }
                }
            } else {
                parse_smiles_batch(&content)
                    .into_iter()
                    .filter_map(|(_, _, r)| r.ok())
                    .collect()
            };

            for (idx, mol) in parsed_mols.into_iter().enumerate() {
                let Ok(smi) = mol.try_to_canonical_smiles() else {
                    continue;
                };
                let name = mol
                    .name
                    .unwrap_or_else(|| format!("Mol_{}_{}", loaded_files + 1, idx + 1));
                combined_lines.push(format!("{smi} {name}"));
            }
            loaded_files += 1;
        }

        let input = combined_lines.join("\n");
        self.run_command(Command::EvaluateBatchText { input });
        self.status_msg = format!(
            "Imported {} molecule(s) from {} dropped files",
            self.session.evaluations.len(),
            loaded_files
        );
    }

    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        let wants_kb = ctx.wants_keyboard_input();
        let mut open_palette = false;
        let mut run_batch = false;
        let mut load_examples = false;
        let mut do_undo = false;
        let mut do_redo = false;
        ctx.input(|i| {
            if i.modifiers.ctrl && i.key_pressed(egui::Key::K) {
                open_palette = true;
            }
            if i.modifiers.ctrl && i.key_pressed(egui::Key::Enter) {
                run_batch = true;
            }
            if !wants_kb && i.modifiers.ctrl && i.key_pressed(egui::Key::E) {
                load_examples = true;
            }
            // Only intercept Ctrl+Z / Ctrl+Y for session undo when no TextEdit has focus
            if !wants_kb && i.modifiers.ctrl && i.key_pressed(egui::Key::Z) {
                do_undo = true;
            }
            if !wants_kb && i.modifiers.ctrl && i.key_pressed(egui::Key::Y) {
                do_redo = true;
            }
        });
        if open_palette {
            self.palette_open = !self.palette_open;
            self.palette_query.clear();
        }
        if run_batch {
            let input = self.session.smiles_input.clone();
            self.run_command(Command::EvaluateBatchText { input });
        }
        if load_examples {
            self.run_command(Command::LoadExampleBatch);
        }
        if do_undo {
            self.run_command(Command::Undo);
        }
        if do_redo {
            self.run_command(Command::Redo);
        }

        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        self.import_dropped_files(&dropped);
    }

    fn render_palette(&mut self, ctx: &egui::Context) {
        if !self.palette_open {
            return;
        }
        let mut close = false;
        let mut exec_id: Option<&'static str> = None;
        egui::Window::new("Command Palette (Ctrl+K)")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_TOP, [0.0, 64.0])
            .fixed_size([460.0, 300.0])
            .show(ctx, |ui| {
                let resp = ui.add(
                    egui::TextEdit::singleline(&mut self.palette_query)
                        .hint_text("Type a command...")
                        .desired_width(f32::INFINITY),
                );
                resp.request_focus();
                ui.separator();

                let q = self.palette_query.to_ascii_lowercase();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for desc in CommandRegistry::catalog() {
                        if !q.is_empty()
                            && !desc.title.to_ascii_lowercase().contains(&q)
                            && !desc.category.to_ascii_lowercase().contains(&q)
                        {
                            continue;
                        }
                        ui.horizontal(|ui| {
                            if ui
                                .selectable_label(
                                    false,
                                    format!("[{}] {}", desc.category, desc.title),
                                )
                                .clicked()
                            {
                                exec_id = Some(desc.id);
                                close = true;
                            }
                            if !desc.shortcut.is_empty() {
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        ui.weak(desc.shortcut);
                                    },
                                );
                            }
                        });
                    }
                });
                ui.separator();
                if ui.button("Close (Esc)").clicked()
                    || ui.input(|i| i.key_pressed(egui::Key::Escape))
                {
                    close = true;
                }
            });

        if let Some(id) = exec_id {
            match id {
                "load_examples" => self.run_command(Command::LoadExampleBatch),
                "run_batch" => {
                    let input = self.session.smiles_input.clone();
                    self.run_command(Command::EvaluateBatchText { input });
                }
                "filter_lipinski_0" => self.run_command(Command::FilterBatch {
                    criteria: FilterCriteria {
                        max_lipinski_violations: Some(0),
                        no_pains: true,
                        ..FilterCriteria::default()
                    },
                }),
                "filter_bbb_yes" => self.run_command(Command::FilterBatch {
                    criteria: FilterCriteria {
                        require_bbb: Some(true),
                        ..FilterCriteria::default()
                    },
                }),
                "filter_qed_05" => self.run_command(Command::FilterBatch {
                    criteria: FilterCriteria {
                        min_qed: Some(0.50),
                        no_pains: true,
                        ..FilterCriteria::default()
                    },
                }),
                "filter_cns_mpo_4" => self.run_command(Command::FilterBatch {
                    criteria: FilterCriteria {
                        min_cns_mpo: Some(4.0),
                        require_bbb: Some(true),
                        ..FilterCriteria::default()
                    },
                }),
                "clear_session" => self.run_command(Command::ClearSession),
                "undo" => self.run_command(Command::Undo),
                "redo" => self.run_command(Command::Redo),
                _ => {}
            }
        }
        if close {
            self.palette_open = false;
        }
    }
}

impl eframe::App for AdmetCraftApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Ok(mut guard) = self.async_status.lock()
            && let Some(msg) = guard.take()
        {
            self.status_msg = msg;
        }

        if self.dark_mode {
            ctx.set_visuals(egui::Visuals::dark());
        } else {
            ctx.set_visuals(egui::Visuals::light());
        }

        self.handle_shortcuts(ctx);
        self.render_palette(ctx);

        egui::TopBottomPanel::top("craft_menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.strong("AdmetCraft");
                ui.label(
                    RichText::new(format!("v{}", env!("CARGO_PKG_VERSION")))
                        .small()
                        .weak(),
                );
                ui.separator();

                ui.menu_button("File", |ui| {
                    if ui.button("Open / Import (.smi, .csv, .sdf)...").clicked() {
                        self.browse_import_file();
                        ui.close_menu();
                    }
                    if ui.button("Load 15 Example Drugs (Ctrl+E)").clicked() {
                        self.run_command(Command::LoadExampleBatch);
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.menu_button("Export Report", |ui| {
                        if ui.button("CSV Table (admetcraft_report.csv)").clicked() {
                            self.export(ctx, ExportFormat::Csv, "admetcraft_report.csv");
                            ui.close_menu();
                        }
                        if ui.button("JSON Full (admetcraft_report.json)").clicked() {
                            self.export(ctx, ExportFormat::Json, "admetcraft_report.json");
                            ui.close_menu();
                        }
                        if ui.button("PDF Report (admetcraft_report.pdf)").clicked() {
                            self.export(ctx, ExportFormat::Pdf, "admetcraft_report.pdf");
                            ui.close_menu();
                        }
                        if ui
                            .button("BOILED-Egg SVG (admetcraft_boiled_egg.svg)")
                            .clicked()
                        {
                            self.export(
                                ctx,
                                ExportFormat::SvgBoiledEgg,
                                "admetcraft_boiled_egg.svg",
                            );
                            ui.close_menu();
                        }
                    });
                    ui.separator();
                    if ui.button("Clear Session").clicked() {
                        self.run_command(Command::ClearSession);
                        ui.close_menu();
                    }
                });

                ui.menu_button("Edit", |ui| {
                    if ui
                        .add_enabled(self.session.can_undo(), egui::Button::new("Undo (Ctrl+Z)"))
                        .clicked()
                    {
                        self.run_command(Command::Undo);
                        ui.close_menu();
                    }
                    if ui
                        .add_enabled(self.session.can_redo(), egui::Button::new("Redo (Ctrl+Y)"))
                        .clicked()
                    {
                        self.run_command(Command::Redo);
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Run ADMET Batch (Ctrl+Enter)").clicked() {
                        let input = self.session.smiles_input.clone();
                        self.run_command(Command::EvaluateBatchText { input });
                        ui.close_menu();
                    }
                });

                ui.menu_button("View", |ui| {
                    ui.selectable_value(&mut self.active_tab, MainTab::Cards, "Molecule Cards");
                    ui.selectable_value(
                        &mut self.active_tab,
                        MainTab::Qsar,
                        "QSAR (TDC + 90% CI + SoM + TPP)",
                    );
                    ui.selectable_value(
                        &mut self.active_tab,
                        MainTab::BoiledEgg,
                        "BOILED-Egg Plot",
                    );
                    ui.selectable_value(
                        &mut self.active_tab,
                        MainTab::Table,
                        "Batch Summary Table",
                    );
                    ui.separator();
                    ui.checkbox(&mut self.show_som_overlay, "2D Sites of Metabolism Overlay");
                    ui.checkbox(&mut self.dark_mode, "Dark Theme");
                });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Ctrl+K").clicked() {
                        self.palette_open = true;
                        self.palette_query.clear();
                    }
                    if ui
                        .small_button(if self.dark_mode { "Light" } else { "Dark" })
                        .clicked()
                    {
                        self.dark_mode = !self.dark_mode;
                    }
                });
            });
        });

        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .button(
                        RichText::new("▶ Run!")
                            .strong()
                            .color(Color32::from_rgb(180, 40, 40)),
                    )
                    .clicked()
                {
                    let input = self.session.smiles_input.clone();
                    self.run_command(Command::EvaluateBatchText { input });
                }
                if ui.button("Open").clicked() {
                    self.browse_import_file();
                }
                if ui.button("Examples").clicked() {
                    self.run_command(Command::LoadExampleBatch);
                }
                if ui.button("Clear").clicked() {
                    self.run_command(Command::ClearSession);
                }
                ui.separator();
                if ui
                    .add_enabled(self.session.can_undo(), egui::Button::new("Undo"))
                    .clicked()
                {
                    self.run_command(Command::Undo);
                }
                if ui
                    .add_enabled(self.session.can_redo(), egui::Button::new("Redo"))
                    .clicked()
                {
                    self.run_command(Command::Redo);
                }
                ui.separator();
                ui.menu_button("Export ▾", |ui| {
                    if ui.button("CSV").clicked() {
                        self.export(ctx, ExportFormat::Csv, "admetcraft_report.csv");
                        ui.close_menu();
                    }
                    if ui.button("JSON").clicked() {
                        self.export(ctx, ExportFormat::Json, "admetcraft_report.json");
                        ui.close_menu();
                    }
                    if ui.button("PDF").clicked() {
                        self.export(ctx, ExportFormat::Pdf, "admetcraft_report.pdf");
                        ui.close_menu();
                    }
                    if ui.button("BOILED-Egg SVG").clicked() {
                        self.export(ctx, ExportFormat::SvgBoiledEgg, "admetcraft_boiled_egg.svg");
                        ui.close_menu();
                    }
                });
                ui.separator();
                ui.label("TPP:");
                let prev_tpp = self.session.tpp_preset;
                let mut cur_tpp = prev_tpp;
                egui::ComboBox::from_id_salt("tpp_preset_combo")
                    .selected_text(cur_tpp.as_str())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut cur_tpp,
                            TppPreset::OralSmallMolecule,
                            TppPreset::OralSmallMolecule.as_str(),
                        );
                        ui.selectable_value(
                            &mut cur_tpp,
                            TppPreset::CnsPenetrant,
                            TppPreset::CnsPenetrant.as_str(),
                        );
                        ui.selectable_value(
                            &mut cur_tpp,
                            TppPreset::PeripheralSelective,
                            TppPreset::PeripheralSelective.as_str(),
                        );
                        ui.selectable_value(
                            &mut cur_tpp,
                            TppPreset::OncologyKinase,
                            TppPreset::OncologyKinase.as_str(),
                        );
                    });
                if cur_tpp != prev_tpp {
                    self.session.set_tpp_preset(cur_tpp);
                    self.status_msg = format!("TPP preset -> {}", cur_tpp.as_str());
                }
            });
        });

        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!(
                    "Molecules: {}  |  Errors: {}  |  {}",
                    self.session.evaluations.len(),
                    self.session.failures.len(),
                    self.status_msg
                ));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        RichText::new(format!("AdmetCraft v{}", env!("CARGO_PKG_VERSION")))
                            .small()
                            .strong(),
                    );
                    ui.separator();
                    ui.add(
                        egui::TextEdit::singleline(&mut self.export_dir)
                            .desired_width(180.0)
                            .hint_text("Sandbox dir"),
                    );
                    ui.label("Sandbox dir:");
                });
            });
        });

        egui::SidePanel::left("input_panel")
            .resizable(true)
            .default_width(360.0)
            .min_width(280.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.strong("2D Molecular Sketcher");
                    self.sketcher.ui(ui, self.dark_mode);
                    ui.horizontal(|ui| {
                        ui.label("Sketcher SMILES:");
                        ui.monospace(&self.sketcher.cached_smiles);
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Add to SMILES List >>").clicked()
                            && !self.sketcher.cached_smiles.is_empty()
                        {
                            let line = format!(
                                "{} Sketched_{}",
                                self.sketcher.cached_smiles,
                                self.session.evaluations.len() + 1
                            );
                            if !self.session.smiles_input.is_empty()
                                && !self.session.smiles_input.ends_with('\n')
                            {
                                self.session.smiles_input.push('\n');
                            }
                            self.session.smiles_input.push_str(&line);
                            let input = self.session.smiles_input.clone();
                            self.run_command(Command::EvaluateBatchText { input });
                        }
                        if ui.button("Clear Sketcher").clicked() {
                            self.sketcher.clear();
                        }
                    });

                    ui.separator();
                    ui.strong("SMILES List (SMILES [Optional Name])");
                    ui.add(
                        egui::TextEdit::multiline(&mut self.session.smiles_input)
                            .desired_rows(9)
                            .desired_width(f32::INFINITY)
                            .font(egui::TextStyle::Monospace),
                    );

                    ui.separator();
                    ui.strong("Import .smi / .csv / .sdf (sandboxed or drag & drop)");
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.import_path)
                                .desired_width(165.0)
                                .hint_text("molecules.smi"),
                        );
                        if ui.button("Load").clicked() && !self.import_path.trim().is_empty() {
                            let p = PathBuf::from(self.import_path.trim());
                            self.import_file_path(&p, false);
                        }
                        if ui.button("Browse...").clicked() {
                            self.browse_import_file();
                        }
                    });

                    if !self.session.failures.is_empty() {
                        ui.separator();
                        ui.colored_label(
                            Color32::from_rgb(200, 60, 60),
                            format!("Parse warnings ({}):", self.session.failures.len()),
                        );
                        for f in &self.session.failures {
                            ui.small(format!("Line {}: {}", f.line, f.error));
                        }
                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut self.active_tab, MainTab::Cards, "Molecule Cards");
                ui.selectable_value(
                    &mut self.active_tab,
                    MainTab::Qsar,
                    "QSAR (TDC + 90% CI + SoM + TPP)",
                );
                ui.selectable_value(&mut self.active_tab, MainTab::BoiledEgg, "BOILED-Egg Plot");
                ui.selectable_value(&mut self.active_tab, MainTab::Table, "Batch Summary Table");
                ui.separator();
                if ui.small_button("Filter: Lipinski 0").clicked() {
                    self.run_command(Command::FilterBatch {
                        criteria: FilterCriteria {
                            max_lipinski_violations: Some(0),
                            ..FilterCriteria::default()
                        },
                    });
                }
                if ui.small_button("Filter: BBB+").clicked() {
                    self.run_command(Command::FilterBatch {
                        criteria: FilterCriteria {
                            require_bbb: Some(true),
                            ..FilterCriteria::default()
                        },
                    });
                }
                if ui.small_button("Filter: No PAINS").clicked() {
                    self.run_command(Command::FilterBatch {
                        criteria: FilterCriteria {
                            no_pains: true,
                            ..FilterCriteria::default()
                        },
                    });
                }
                if ui.small_button("Filter: QED>=0.5").clicked() {
                    self.run_command(Command::FilterBatch {
                        criteria: FilterCriteria {
                            min_qed: Some(0.50),
                            ..FilterCriteria::default()
                        },
                    });
                }
                if ui.small_button("Filter: CNS MPO>=4").clicked() {
                    self.run_command(Command::FilterBatch {
                        criteria: FilterCriteria {
                            min_cns_mpo: Some(4.0),
                            ..FilterCriteria::default()
                        },
                    });
                }
                if ui.small_button("Filter: Low/Mod hERG").clicked() {
                    self.run_command(Command::FilterBatch {
                        criteria: FilterCriteria {
                            no_high_herg: true,
                            ..FilterCriteria::default()
                        },
                    });
                }
            });
            ui.separator();

            if self.session.evaluations.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label("No molecules evaluated yet. Click 'Examples' or enter SMILES and press 'Run!'.");
                });
                return;
            }

            match self.active_tab {
                MainTab::Cards => self.render_cards_tab(ui),
                MainTab::Qsar => self.render_qsar_tab(ui),
                MainTab::BoiledEgg => {
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.show_egg_names, "Show Molecule Names");
                        ui.weak("(Hover a point for 2D structure; click to inspect its card)");
                    });
                    if let Some(clicked) = draw_boiled_egg_interactive(
                        ui,
                        &self.session.evaluations,
                        &self.mol_cache,
                        self.show_egg_names,
                        self.session.selected_index,
                        self.dark_mode,
                    ) {
                        self.session.selected_index = Some(clicked);
                        if let Some(ev) = self.session.evaluations.get(clicked) {
                            let _ = self.sketcher.load_smiles(&ev.canonical_smiles);
                        }
                        self.active_tab = MainTab::Cards;
                    }
                }
                MainTab::Table => self.render_table_tab(ui),
            }
        });
    }
}

impl AdmetCraftApp {
    fn render_cards_tab(&mut self, ui: &mut egui::Ui) {
        let mut next_selected: Option<usize> = None;
        egui::ScrollArea::horizontal()
            .id_salt("mol_chips")
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    for (idx, ev) in self.session.evaluations.iter().enumerate() {
                        let is_sel = self.session.selected_index == Some(idx);
                        if ui.selectable_label(is_sel, &ev.name).clicked() {
                            next_selected = Some(idx);
                        }
                    }
                });
            });
        if let Some(idx) = next_selected {
            self.session.selected_index = Some(idx);
            if let Some(ev) = self.session.evaluations.get(idx) {
                let _ = self.sketcher.load_smiles(&ev.canonical_smiles);
            }
        }
        ui.separator();

        let sel_idx = self.session.selected_index.unwrap_or(0);
        let Some(eval) = self.session.evaluations.get(sel_idx).cloned() else {
            return;
        };

        egui::ScrollArea::vertical()
            .id_salt("card_body")
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.heading(&eval.name);
                    ui.label(format!("({})", eval.physicochemical.formula));
                    if ui.small_button("Load into Sketcher").clicked() {
                        let _ = self.sketcher.load_smiles(&eval.canonical_smiles);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Canonical SMILES:");
                    ui.monospace(&eval.canonical_smiles);
                });
                if !eval.warnings.is_empty() {
                    for w in &eval.warnings {
                        ui.colored_label(Color32::from_rgb(180, 95, 10), format!("Note: {w}"));
                    }
                }
                ui.add_space(6.0);

                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.strong("2D Structure");
                            ui.checkbox(&mut self.show_som_overlay, "SoM Hotspots");
                        });
                        let (rect, _) =
                            ui.allocate_exact_size(Vec2::new(260.0, 220.0), Sense::hover());
                        let bg = if self.dark_mode {
                            Color32::from_rgb(30, 32, 36)
                        } else {
                            Color32::WHITE
                        };
                        ui.painter_at(rect).rect_filled(rect, 4.0, bg);
                        if let Some(mol) = self.mol_cache.get(&eval.canonical_smiles) {
                            if self.show_som_overlay && !eval.metabolism_sites.is_empty() {
                                draw_cached_molecule_2d_with_som(
                                    &ui.painter_at(rect),
                                    mol,
                                    rect,
                                    self.dark_mode,
                                    &eval.metabolism_sites,
                                );
                            } else {
                                draw_cached_molecule_2d(
                                    &ui.painter_at(rect),
                                    mol,
                                    rect,
                                    self.dark_mode,
                                );
                            }
                        }
                    });
                    ui.add_space(16.0);
                    ui.vertical(|ui| {
                        ui.strong("Bioavailability Radar");
                        draw_bioavailability_radar(ui, &eval, 220.0, self.dark_mode);
                    });
                });

                ui.add_space(6.0);
                ui.horizontal_wrapped(|ui| {
                    let qed_col = if eval.druglikeness.qed >= 0.67 {
                        Color32::from_rgb(35, 145, 70)
                    } else if eval.druglikeness.qed >= 0.45 {
                        Color32::from_rgb(185, 125, 20)
                    } else {
                        Color32::from_rgb(195, 50, 50)
                    };
                    let mpo_col = if eval.druglikeness.cns_mpo >= 4.0 {
                        Color32::from_rgb(35, 145, 70)
                    } else {
                        Color32::from_rgb(120, 120, 130)
                    };
                    let herg_col = match eval.safety.herg_risk {
                        admet_calc::HergRisk::Low => Color32::from_rgb(35, 145, 70),
                        admet_calc::HergRisk::Moderate => Color32::from_rgb(185, 125, 20),
                        admet_calc::HergRisk::High => Color32::from_rgb(195, 50, 50),
                    };
                    ui.colored_label(
                        qed_col,
                        RichText::new(format!("QED: {:.3}", eval.druglikeness.qed)).strong(),
                    );
                    ui.separator();
                    ui.colored_label(
                        mpo_col,
                        RichText::new(format!("CNS MPO: {:.2}/6.0", eval.druglikeness.cns_mpo))
                            .strong(),
                    );
                    ui.separator();
                    ui.label(
                        RichText::new(format!(
                            "LogD7.4: {:.2} ({})",
                            eval.ionization.logd_ph74,
                            eval.ionization.ionization_class.as_str()
                        ))
                        .strong(),
                    );
                    ui.separator();
                    ui.colored_label(
                        herg_col,
                        RichText::new(format!("hERG: {}", eval.safety.herg_risk.as_str())).strong(),
                    );
                    ui.separator();
                    ui.label(format!(
                        "PAINS: {} | Brenk: {} | AMES: {} | DILI: {}",
                        eval.medicinal_chemistry.pains_alerts.len(),
                        eval.medicinal_chemistry.brenk_alerts.len(),
                        eval.safety.ames_alerts.len(),
                        eval.safety.dili_alerts.len()
                    ));
                    if let Some(tpp) = &eval.tpp_mpo {
                        ui.separator();
                        let tc = if tpp.decision == "Advance" {
                            Color32::from_rgb(35, 145, 70)
                        } else if tpp.decision == "Optimize" {
                            Color32::from_rgb(185, 125, 20)
                        } else {
                            Color32::from_rgb(195, 50, 50)
                        };
                        ui.colored_label(
                            tc,
                            RichText::new(format!("TPP MPO: {:.2} ({})", tpp.score, tpp.decision))
                                .strong(),
                        );
                    }
                    if let Some(qsar) = &eval.qsar {
                        ui.separator();
                        let adc = match qsar.overall_ad.status {
                            ApplicabilityDomainStatus::InDomain => Color32::from_rgb(35, 145, 70),
                            ApplicabilityDomainStatus::Borderline => {
                                Color32::from_rgb(185, 125, 20)
                            }
                            ApplicabilityDomainStatus::OutOfDomain => {
                                Color32::from_rgb(195, 50, 50)
                            }
                        };
                        ui.colored_label(
                            adc,
                            RichText::new(format!(
                                "AD: {} (T={:.2})",
                                qsar.overall_ad.status.as_str(),
                                qsar.overall_ad.max_tanimoto
                            ))
                            .strong(),
                        );
                    }
                });

                ui.add_space(8.0);
                render_admet_sections(ui, &eval);
                ui.add_space(6.0);
                egui::CollapsingHeader::new("Human PK / Minimal-PBPK Dose Simulator (v1.2)")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Target IC50 / Ki (nM):");
                            ui.add(
                                egui::DragValue::new(&mut self.pk_potency_nm)
                                    .speed(0.5)
                                    .range(0.1..=5000.0),
                            );
                            ui.separator();
                            ui.label("Route:");
                            ui.selectable_value(&mut self.pk_route, PkRoute::Oral, "Oral");
                            ui.selectable_value(&mut self.pk_route, PkRoute::Iv, "IV");
                            ui.selectable_value(&mut self.pk_route, PkRoute::Sc, "SC");
                        });
                        let pk_rep = simulate_human_pk_dose(
                            &eval,
                            &PkSimulationConfig {
                                target_potency_nm: self.pk_potency_nm,
                                route: self.pk_route,
                                ..Default::default()
                            },
                        );
                        kv_grid(
                            ui,
                            "pbpk_grid",
                            &[
                                ("Recommended Regimen", pk_rep.recommended_regimen.clone()),
                                ("BCS Classification", pk_rep.bcs_class.as_str().to_string()),
                                (
                                    "Oral Bioavailability (F)",
                                    format!(
                                        "{:.1}% (Fa={:.2}, Fg={:.2}, Fh={:.2})",
                                        pk_rep.oral_bioavailability_f * 100.0,
                                        pk_rep.fa,
                                        pk_rep.fg,
                                        pk_rep.fh
                                    ),
                                ),
                                (
                                    "CL_tot / V_d,ss / t1/2",
                                    format!(
                                        "{:.2} mL/min/kg | {:.2} L/kg | {:.1} h (fup={:.3})",
                                        pk_rep.cl_total_ml_min_kg,
                                        pk_rep.vdss_l_kg,
                                        pk_rep.t_half_ivive_h,
                                        pk_rep.fup
                                    ),
                                ),
                                (
                                    "QD Dose (90% CI)",
                                    format!(
                                        "{:.1} mg/d ({:.0}-{:.0} mg) | Cmax,u={:.1} nM | hERG Margin={:.0}x",
                                        pk_rep.qd.daily_dose_mg,
                                        pk_rep.qd.daily_dose_ci90_mg[0],
                                        pk_rep.qd.daily_dose_ci90_mg[1],
                                        pk_rep.qd.cmax_u_nm,
                                        pk_rep.qd.herg_safety_margin
                                    ),
                                ),
                                (
                                    "BID Dose (90% CI)",
                                    format!(
                                        "{:.1} mg q12h ({:.0}-{:.0} mg/d) | {}",
                                        pk_rep.bid.dose_mg_per_admin,
                                        pk_rep.bid.daily_dose_ci90_mg[0],
                                        pk_rep.bid.daily_dose_ci90_mg[1],
                                        pk_rep.bid.feasibility
                                    ),
                                ),
                            ],
                        );
                        // Compact live egui sparkline of steady-state C_u(t) vs IC50
                        if pk_rep.steady_state_curve.len() >= 2 {
                            let (rect, _) =
                                ui.allocate_exact_size(Vec2::new(460.0, 120.0), Sense::hover());
                            let painter = ui.painter_at(rect);
                            let bg = if self.dark_mode {
                                Color32::from_rgb(24, 28, 36)
                            } else {
                                Color32::from_rgb(248, 250, 252)
                            };
                            painter.rect_filled(rect, 4.0, bg);
                            let max_y = pk_rep
                                .steady_state_curve
                                .iter()
                                .map(|p| p.cu_p95_nm.max(p.cu_median_nm))
                                .fold(self.pk_potency_nm * 2.0, f64::max)
                                * 1.1;
                            let map_pt = |t: f64, c: f64| -> egui::Pos2 {
                                let x = rect.left() + 8.0 + ((t / 48.0).clamp(0.0, 1.0) as f32) * (rect.width() - 16.0);
                                let y = rect.bottom() - 8.0 - ((c / max_y.max(1.0)).clamp(0.0, 1.0) as f32) * (rect.height() - 16.0);
                                egui::pos2(x, y)
                            };
                            let y_ic50 = map_pt(0.0, self.pk_potency_nm).y;
                            painter.line_segment(
                                [egui::pos2(rect.left() + 8.0, y_ic50), egui::pos2(rect.right() - 8.0, y_ic50)],
                                egui::Stroke::new(1.2_f32, Color32::from_rgb(22, 163, 74)),
                            );
                            let pts: Vec<egui::Pos2> = pk_rep
                                .steady_state_curve
                                .iter()
                                .map(|p| map_pt(p.time_h, p.cu_median_nm))
                                .collect();
                            painter.add(egui::Shape::line(
                                pts,
                                egui::Stroke::new(2.0_f32, Color32::from_rgb(37, 99, 235)),
                            ));
                        }
                        for w in &pk_rep.warnings {
                            ui.colored_label(Color32::from_rgb(185, 110, 20), format!("• {w}"));
                        }
                    });
            });
    }

    fn render_qsar_tab(&mut self, ui: &mut egui::Ui) {
        let mut next_selected: Option<usize> = None;
        egui::ScrollArea::horizontal()
            .id_salt("qsar_mol_chips")
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    for (idx, ev) in self.session.evaluations.iter().enumerate() {
                        let is_sel = self.session.selected_index == Some(idx);
                        if ui.selectable_label(is_sel, &ev.name).clicked() {
                            next_selected = Some(idx);
                        }
                    }
                });
            });
        if let Some(idx) = next_selected {
            self.session.selected_index = Some(idx);
            if let Some(ev) = self.session.evaluations.get(idx) {
                let _ = self.sketcher.load_smiles(&ev.canonical_smiles);
            }
        }
        ui.separator();

        let sel_idx = self.session.selected_index.unwrap_or(0);
        let Some(eval) = self.session.evaluations.get(sel_idx).cloned() else {
            return;
        };

        egui::ScrollArea::vertical()
            .id_salt("qsar_body")
            .show(ui, |ui| {
                // 1. TPP MPO Scorecard
                if let Some(tpp) = &eval.tpp_mpo {
                    let dec_col = if tpp.decision == "Advance" {
                        Color32::from_rgb(35, 145, 70)
                    } else if tpp.decision == "Optimize" {
                        Color32::from_rgb(185, 125, 20)
                    } else {
                        Color32::from_rgb(195, 50, 50)
                    };
                    ui.horizontal(|ui| {
                        ui.heading(format!("{} — TPP MPO", eval.name));
                        ui.colored_label(
                            dec_col,
                            RichText::new(format!("{:.3} ({})", tpp.score, tpp.decision))
                                .size(17.0)
                                .strong(),
                        );
                        ui.weak(format!("[{}]", tpp.preset.as_str()));
                    });
                    for hs in &tpp.hard_stops {
                        ui.colored_label(
                            Color32::from_rgb(200, 40, 40),
                            RichText::new(format!("HARD STOP: {hs}")).strong(),
                        );
                    }
                    ui.add_space(4.0);
                    egui::Grid::new("tpp_grid")
                        .striped(true)
                        .spacing([12.0, 4.0])
                        .show(ui, |ui| {
                            for h in ["Objective", "Observed", "Target Spec", "Desirability", "Status"] {
                                ui.strong(h);
                            }
                            ui.end_row();
                            for c in &tpp.criteria {
                                ui.label(&c.name);
                                ui.monospace(&c.observed);
                                ui.weak(&c.target_spec);
                                ui.label(format!("{:.2}", c.desirability));
                                let sc = match c.status {
                                    TppStatus::Pass => Color32::from_rgb(35, 145, 70),
                                    TppStatus::Warn => Color32::from_rgb(185, 125, 20),
                                    TppStatus::Fail => Color32::from_rgb(195, 50, 50),
                                };
                                ui.colored_label(sc, RichText::new(c.status.as_str()).strong());
                                ui.end_row();
                            }
                        });
                    ui.separator();
                }

                // 2. Quantitative TDC QSAR Panel (20 endpoints + 90% Conformal CI + AD)
                if let Some(qsar) = &eval.qsar {
                    let adc = match qsar.overall_ad.status {
                        ApplicabilityDomainStatus::InDomain => Color32::from_rgb(35, 145, 70),
                        ApplicabilityDomainStatus::Borderline => Color32::from_rgb(185, 125, 20),
                        ApplicabilityDomainStatus::OutOfDomain => Color32::from_rgb(195, 50, 50),
                    };
                    ui.horizontal_wrapped(|ui| {
                        ui.strong(format!(
                            "Quantitative TDC QSAR Panel ({} endpoints, v{})",
                            qsar.endpoints.len(),
                            qsar.model_pack_version
                        ));
                        ui.separator();
                        ui.colored_label(
                            adc,
                            RichText::new(format!(
                                "Global AD: {} (maxSim={:.3}, top5={:.3}, Z={:.2})",
                                qsar.overall_ad.status.as_str(),
                                qsar.overall_ad.max_tanimoto,
                                qsar.overall_ad.top5_tanimoto,
                                qsar.overall_ad.descriptor_z_rms
                            ))
                            .strong(),
                        );
                    });
                    ui.add_space(4.0);
                    egui::Grid::new("qsar_endpoints_grid")
                        .striped(true)
                        .spacing([12.0, 4.0])
                        .show(ui, |ui| {
                            for h in [
                                "Endpoint",
                                "Category",
                                "Prediction (90% Conformal CI)",
                                "Conformal Call",
                                "AD Status",
                                "Scaffold Val.",
                            ] {
                                ui.strong(h);
                            }
                            ui.end_row();
                            for ep in &qsar.endpoints {
                                ui.label(&ep.display_name);
                                ui.weak(&ep.category);
                                match ep.task {
                                    QsarTaskKind::BinaryClassification => {
                                        let col = if ep.category == "Toxicity"
                                            || ep.category == "Metabolism"
                                        {
                                            if ep.value >= 0.65 {
                                                Color32::from_rgb(195, 50, 50)
                                            } else if ep.value >= 0.40 {
                                                Color32::from_rgb(185, 125, 20)
                                            } else {
                                                Color32::from_rgb(35, 145, 70)
                                            }
                                        } else {
                                            ui.visuals().text_color()
                                        };
                                        ui.colored_label(
                                            col,
                                            RichText::new(format!(
                                                "P = {:.3} [{:.2}..{:.2}]",
                                                ep.value, ep.ci90_low, ep.ci90_high
                                            ))
                                            .monospace(),
                                        );
                                    }
                                    QsarTaskKind::Regression => {
                                        ui.monospace(&ep.conformal_call);
                                    }
                                }
                                ui.label(match ep.task {
                                    QsarTaskKind::BinaryClassification => ep.conformal_call.as_str(),
                                    QsarTaskKind::Regression => "90% Split-Conformal",
                                });
                                let ep_adc = match ep.applicability_domain.status {
                                    ApplicabilityDomainStatus::InDomain => {
                                        Color32::from_rgb(35, 145, 70)
                                    }
                                    ApplicabilityDomainStatus::Borderline => {
                                        Color32::from_rgb(185, 125, 20)
                                    }
                                    ApplicabilityDomainStatus::OutOfDomain => {
                                        Color32::from_rgb(195, 50, 50)
                                    }
                                };
                                ui.colored_label(
                                    ep_adc,
                                    format!(
                                        "{} ({:.2})",
                                        ep.applicability_domain.status.as_str(),
                                        ep.applicability_domain.max_tanimoto
                                    ),
                                );
                                ui.weak(format!(
                                    "{} {:.3} (Cov {:.0}%)",
                                    ep.test_metric_name,
                                    ep.test_metric_value,
                                    ep.test_conformal_coverage * 100.0
                                ));
                                ui.end_row();
                            }
                        });
                } else {
                    ui.colored_label(
                        Color32::from_rgb(185, 125, 20),
                        "No local TDC QSAR model pack found. Run `admetcraft-cli train-qsar` to generate it in CRAFT_DATA_DIR/models/admetcraft_v1.",
                    );
                }

                ui.separator();
                // 3. Atomic Sites of Metabolism (SoM) + 2D visual map
                ui.strong(format!(
                    "Atomic Sites of Metabolism — CYP450 & Phase II ({} hotspot(s))",
                    eval.metabolism_sites.len()
                ));
                ui.horizontal(|ui| {
                    let (rect, _) = ui.allocate_exact_size(Vec2::new(260.0, 210.0), Sense::hover());
                    let bg = if self.dark_mode {
                        Color32::from_rgb(30, 32, 36)
                    } else {
                        Color32::WHITE
                    };
                    ui.painter_at(rect).rect_filled(rect, 4.0, bg);
                    if let Some(mol) = self.mol_cache.get(&eval.canonical_smiles) {
                        draw_cached_molecule_2d_with_som(
                            &ui.painter_at(rect),
                            mol,
                            rect,
                            self.dark_mode,
                            &eval.metabolism_sites,
                        );
                    }
                    ui.add_space(12.0);
                    ui.vertical(|ui| {
                        if eval.metabolism_sites.is_empty() {
                            ui.weak("No high-lability metabolic soft spots detected.");
                        } else {
                            egui::Grid::new("som_grid")
                                .striped(true)
                                .spacing([10.0, 4.0])
                                .show(ui, |ui| {
                                    for h in ["Atom", "Isoform", "Reaction", "Score", "MedChem Fix"] {
                                        ui.strong(h);
                                    }
                                    ui.end_row();
                                    for s in &eval.metabolism_sites {
                                        ui.monospace(format!("{}{}", s.atom_symbol, s.atom_index + 1));
                                        ui.label(&s.isoform);
                                        ui.label(&s.reaction);
                                        let rc = match s.risk {
                                            MetabolismRisk::High => Color32::from_rgb(195, 50, 50),
                                            MetabolismRisk::Moderate => Color32::from_rgb(185, 125, 20),
                                            MetabolismRisk::Low => Color32::from_rgb(120, 120, 130),
                                        };
                                        ui.colored_label(
                                            rc,
                                            RichText::new(format!("{:.2} ({})", s.lability_score, s.risk.as_str()))
                                                .strong(),
                                        );
                                        ui.weak(&s.medchem_fix);
                                        ui.end_row();
                                    }
                                });
                        }
                    });
                });

                // 4. Tox21 Cellular & Endocrine Disruption Pathways (12 assays)
                if let Some(tox21) = &eval.tox21_summary {
                    ui.separator();
                    let n_alerts = tox21.endocrine_alerts.len() + tox21.stress_alerts.len();
                    egui::CollapsingHeader::new(format!(
                        "Tox21 Cellular & Endocrine Pathways ({} assays, {} alert(s), max P={:.2})",
                        tox21.assays_evaluated, n_alerts, tox21.max_probability
                    ))
                    .default_open(n_alerts > 0)
                    .show(ui, |ui| {
                        if n_alerts == 0 {
                            ui.colored_label(
                                Color32::from_rgb(45, 145, 75),
                                "No Nuclear Receptor (NR) or Stress Response (SR) alerts above P=0.50.",
                            );
                        } else {
                            for a in &tox21.endocrine_alerts {
                                ui.colored_label(Color32::from_rgb(195, 50, 50), format!("NR Alert: {a}"));
                            }
                            for a in &tox21.stress_alerts {
                                ui.colored_label(Color32::from_rgb(195, 50, 50), format!("SR Alert: {a}"));
                            }
                        }
                    });
                }

                // 5. SEA Off-Target Safety Pharmacology (Bowes Safety-20)
                if let Some(sea) = &eval.off_targets {
                    ui.separator();
                    egui::CollapsingHeader::new(format!(
                        "SEA Off-Target Fishing — {} ({} High, {} Moderate / {} targets)",
                        sea.panel_version,
                        sea.high_risk_count,
                        sea.moderate_risk_count,
                        sea.targets_evaluated
                    ))
                    .default_open(sea.high_risk_count > 0)
                    .show(ui, |ui| {
                        if sea.hits.is_empty() {
                            ui.colored_label(
                                Color32::from_rgb(45, 145, 75),
                                "No significant off-target similarity alerts detected across Bowes Safety-20.",
                            );
                        } else {
                            egui::Grid::new("sea_offtarget_grid")
                                .striped(true)
                                .spacing([10.0, 4.0])
                                .show(ui, |ui| {
                                    for h in ["Target", "Risk", "Max Tc", "Z-Score", "Nearest Drug", "Organ Liability"] {
                                        ui.strong(h);
                                    }
                                    ui.end_row();
                                    for hit in &sea.hits {
                                        ui.monospace(&hit.target_id);
                                        let col = match hit.risk {
                                            admet_calc::OffTargetRisk::High => Color32::from_rgb(195, 50, 50),
                                            admet_calc::OffTargetRisk::Moderate => Color32::from_rgb(185, 125, 20),
                                            admet_calc::OffTargetRisk::Low => Color32::from_rgb(120, 120, 130),
                                        };
                                        ui.colored_label(col, RichText::new(hit.risk.as_str()).strong());
                                        ui.monospace(format!("{:.3}", hit.max_tanimoto));
                                        ui.monospace(format!("{:+.2}", hit.z_score));
                                        ui.label(&hit.nearest_ligand);
                                        ui.weak(&hit.organ_liability);
                                        ui.end_row();
                                    }
                                });
                        }
                    });
                }

                // 6. Phase I/II Metabolite Guard (Biotransformation -> Re-Screening)
                if let Some(mg) = &eval.metabolite_guard {
                    ui.separator();
                    egui::CollapsingHeader::new(format!(
                        "Metabolite Guard — {} generated ({} bioactivation, {} escalation)",
                        mg.total_generated, mg.bioactivation_alerts, mg.escalation_count
                    ))
                    .default_open(mg.bioactivation_alerts > 0 || mg.escalation_count > 0)
                    .show(ui, |ui| {
                        if mg.metabolites.is_empty() {
                            ui.weak("No major Phase I/II biotransformation products generated.");
                        } else {
                            egui::Grid::new("metabolite_guard_grid")
                                .striped(true)
                                .spacing([10.0, 4.0])
                                .show(ui, |ui| {
                                    for h in ["#", "Phase", "Reaction", "Bioact.", "DILI / AMES", "Metabolite SMILES"] {
                                        ui.strong(h);
                                    }
                                    ui.end_row();
                                    for m in &mg.metabolites {
                                        ui.monospace(format!("{}", m.rank));
                                        ui.label(m.phase.as_str());
                                        ui.label(&m.reaction);
                                        if m.reactive_bioactivation {
                                            ui.colored_label(
                                                Color32::from_rgb(195, 50, 50),
                                                RichText::new("ALERT").strong(),
                                            );
                                        } else if m.toxicity_escalation {
                                            ui.colored_label(
                                                Color32::from_rgb(185, 125, 20),
                                                RichText::new("ESC").strong(),
                                            );
                                        } else {
                                            ui.colored_label(Color32::from_rgb(45, 145, 75), "OK");
                                        }
                                        let dp = m.dili_prob.map_or_else(|| "-".to_string(), |v| format!("{v:.2}"));
                                        let ap = m.ames_prob.map_or_else(|| "-".to_string(), |v| format!("{v:.2}"));
                                        ui.monospace(format!("{dp} / {ap}"));
                                        ui.monospace(&m.smiles);
                                        ui.end_row();
                                    }
                                });
                        }
                    });
                }
            });
    }

    fn render_table_tab(&mut self, ui: &mut egui::Ui) {
        let mut picked: Option<usize> = None;
        egui::ScrollArea::both().show(ui, |ui| {
            egui::Grid::new("batch_summary_grid")
                .striped(true)
                .spacing([12.0, 6.0])
                .show(ui, |ui| {
                    for h in [
                        "Molecule", "Formula", "MW", "TPSA", "cLogP", "LogD7.4", "State", "ESOL",
                        "QED", "CNS MPO", "GI", "BBB", "Lipinski", "PAINS", "Brenk", "hERG",
                        "AMES", "DILI", "SA",
                    ] {
                        ui.strong(h);
                    }
                    ui.end_row();

                    for (idx, e) in self.session.evaluations.iter().enumerate() {
                        if ui
                            .selectable_label(self.session.selected_index == Some(idx), &e.name)
                            .clicked()
                        {
                            picked = Some(idx);
                        }
                        ui.label(&e.physicochemical.formula);
                        ui.label(format!("{:.2}", e.physicochemical.molecular_weight));
                        ui.label(format!("{:.2}", e.physicochemical.tpsa));
                        ui.label(format!("{:.2}", e.lipophilicity.consensus_log_p));
                        ui.label(format!("{:.2}", e.ionization.logd_ph74));
                        ui.label(e.ionization.ionization_class.as_str());
                        ui.label(format!("{:.2}", e.water_solubility.esol.log_s));
                        ui.label(format!("{:.3}", e.druglikeness.qed));
                        ui.label(format!("{:.2}", e.druglikeness.cns_mpo));
                        ui.label(e.pharmacokinetics.gi_absorption.as_str());
                        ui.label(yes_no(e.pharmacokinetics.bbb_permeant));
                        ui.label(format!(
                            "{} ({})",
                            yes_no(e.druglikeness.lipinski.pass),
                            e.druglikeness.lipinski.violations
                        ));
                        ui.label(format!(
                            "{}/{}",
                            e.medicinal_chemistry.pains_alerts.len(),
                            e.medicinal_chemistry.pains_catalog_size
                        ));
                        ui.label(format!(
                            "{}/{}",
                            e.medicinal_chemistry.brenk_alerts.len(),
                            e.medicinal_chemistry.brenk_catalog_size
                        ));
                        ui.label(e.safety.herg_risk.as_str());
                        ui.label(e.safety.ames_alerts.len().to_string());
                        ui.label(e.safety.dili_alerts.len().to_string());
                        ui.label(format!(
                            "{:.2}",
                            e.medicinal_chemistry.synthetic_accessibility
                        ));
                        ui.end_row();
                    }
                });
        });
        if let Some(idx) = picked {
            self.session.selected_index = Some(idx);
            if let Some(ev) = self.session.evaluations.get(idx) {
                let _ = self.sketcher.load_smiles(&ev.canonical_smiles);
            }
            self.active_tab = MainTab::Cards;
        }
    }
}

fn render_admet_sections(ui: &mut egui::Ui, e: &MoleculeEvaluation) {
    let p = &e.physicochemical;
    let l = &e.lipophilicity;
    let ion = &e.ionization;
    let w = &e.water_solubility;
    let pk = &e.pharmacokinetics;
    let d = &e.druglikeness;
    let mc = &e.medicinal_chemistry;
    let s = &e.safety;

    egui::CollapsingHeader::new("Physicochemical Properties")
        .default_open(true)
        .show(ui, |ui| {
            kv_grid(
                ui,
                "phys_grid",
                &[
                    ("Formula", p.formula.clone()),
                    (
                        "Molecular weight",
                        format!("{:.2} g/mol", p.molecular_weight),
                    ),
                    (
                        "Exact mass (monoisotopic)",
                        format!("{:.4} Da", p.exact_mass),
                    ),
                    ("Num. heavy atoms", p.heavy_atoms.to_string()),
                    ("Num. arom. heavy atoms", p.aromatic_heavy_atoms.to_string()),
                    (
                        "Rings / Aromatic rings",
                        format!("{} / {}", p.num_rings, p.num_aromatic_rings),
                    ),
                    ("DBE (unsaturation)", format!("{:.1}", p.dbe)),
                    ("Fraction Csp3", format!("{:.2}", p.fraction_csp3)),
                    ("Num. rotatable bonds", p.rotatable_bonds.to_string()),
                    ("Num. H-bond acceptors", p.h_bond_acceptors.to_string()),
                    ("Num. H-bond donors", p.h_bond_donors.to_string()),
                    ("Molar Refractivity", format!("{:.2}", p.molar_refractivity)),
                    ("TPSA (S & P, SwissADME)", format!("{:.2} A^2", p.tpsa)),
                    (
                        "TPSA (N & O only, RDKit)",
                        format!("{:.2} A^2", p.tpsa_no_only),
                    ),
                ],
            );
        });

    egui::CollapsingHeader::new("Lipophilicity & Ionization (pH 7.4)")
        .default_open(true)
        .show(ui, |ui| {
            let pka_a = ion
                .strongest_acidic_pka
                .map(|v| format!("{v:.1}"))
                .unwrap_or_else(|| "None".to_string());
            let pka_b = ion
                .strongest_basic_pka
                .map(|v| format!("{v:.1}"))
                .unwrap_or_else(|| "None".to_string());
            kv_grid(
                ui,
                "lipo_grid",
                &[
                    ("Log Po/w (WLOGP)", format!("{:.2}", l.wlogp)),
                    ("Log Po/w (XLOGP3 heur.)", format!("{:.2}", l.xlogp3)),
                    ("Log Po/w (MLOGP)", format!("{:.2}", l.mlogp)),
                    (
                        "Solvation est. (exp.)",
                        format!("{:.2}", l.solvation_logp_est),
                    ),
                    (
                        "Topological est. (exp.)",
                        format!("{:.2}", l.topological_logp_est),
                    ),
                    (
                        "Consensus Log Po/w (3-model)",
                        format!("{:.2}", l.consensus_log_p),
                    ),
                    (
                        "LogD @ pH 7.4 (Henderson-Hasselbalch)",
                        format!("{:.2}", ion.logd_ph74),
                    ),
                    (
                        "Ionization state @ pH 7.4",
                        format!(
                            "{} (est. charge {:+.2})",
                            ion.ionization_class.as_str(),
                            ion.estimated_charge_ph74
                        ),
                    ),
                    ("Strongest Acidic pKa (heur.)", pka_a),
                    ("Strongest Basic pKa (heur.)", pka_b),
                ],
            );
        });

    egui::CollapsingHeader::new("Water Solubility")
        .default_open(true)
        .show(ui, |ui| {
            kv_grid(
                ui,
                "sol_grid",
                &[
                    (
                        "Log S (ESOL)",
                        format!(
                            "{:.2} | {:.2e} mg/ml; {:.2e} mol/l | {}",
                            w.esol.log_s,
                            w.esol.solubility_mg_ml,
                            w.esol.solubility_mol_l,
                            w.esol.class.as_str()
                        ),
                    ),
                    (
                        "Log S (Ali)",
                        format!(
                            "{:.2} | {:.2e} mg/ml; {:.2e} mol/l | {}",
                            w.ali.log_s,
                            w.ali.solubility_mg_ml,
                            w.ali.solubility_mol_l,
                            w.ali.class.as_str()
                        ),
                    ),
                    (
                        "Log S (Topo exp.)",
                        format!(
                            "{:.2} | {:.2e} mg/ml; {:.2e} mol/l | {}",
                            w.topological_est.log_s,
                            w.topological_est.solubility_mg_ml,
                            w.topological_est.solubility_mol_l,
                            w.topological_est.class.as_str()
                        ),
                    ),
                ],
            );
        });

    egui::CollapsingHeader::new("Pharmacokinetics")
        .default_open(true)
        .show(ui, |ui| {
            kv_grid(
                ui,
                "pk_grid",
                &[
                    ("GI absorption", pk.gi_absorption.as_str().to_string()),
                    ("BBB permeant", yes_no(pk.bbb_permeant).to_string()),
                    (
                        "P-gp substrate (exp. heur.)",
                        yes_no(pk.pgp_substrate).to_string(),
                    ),
                    (
                        "CYP1A2 inhibitor (exp. heur.)",
                        yes_no(pk.cyp1a2_inhibitor).to_string(),
                    ),
                    (
                        "CYP2C19 inhibitor (exp. heur.)",
                        yes_no(pk.cyp2c19_inhibitor).to_string(),
                    ),
                    (
                        "CYP2C9 inhibitor (exp. heur.)",
                        yes_no(pk.cyp2c9_inhibitor).to_string(),
                    ),
                    (
                        "CYP2D6 inhibitor (exp. heur.)",
                        yes_no(pk.cyp2d6_inhibitor).to_string(),
                    ),
                    (
                        "CYP3A4 inhibitor (exp. heur.)",
                        yes_no(pk.cyp3a4_inhibitor).to_string(),
                    ),
                    (
                        "Log Kp (skin permeation)",
                        format!("{:.2} cm/s", pk.log_kp_cm_s),
                    ),
                ],
            );
        });

    egui::CollapsingHeader::new("Druglikeness & Industrial MedChem Rules")
        .default_open(true)
        .show(ui, |ui| {
            kv_grid(
                ui,
                "drug_grid",
                &[
                    ("QED (Bickerton 2012)", format!("{:.3} / 1.000", d.qed)),
                    (
                        "Pfizer CNS MPO (Wager 2010)",
                        format!("{:.2} / 6.00", d.cns_mpo),
                    ),
                    ("Lipinski (Ro5)", fmt_rule(&d.lipinski)),
                    ("Ghose", fmt_rule(&d.ghose)),
                    ("Veber", fmt_rule(&d.veber)),
                    ("Egan", fmt_rule(&d.egan)),
                    ("Muegge", fmt_rule(&d.muegge)),
                    ("Pfizer 3/75 (Toxicity Risk)", fmt_rule(&d.pfizer_3_75)),
                    ("GSK 4/400 (ADMET Risk)", fmt_rule(&d.gsk_4_400)),
                    ("Golden Triangle (MW/LogD)", fmt_rule(&d.golden_triangle)),
                    ("Rule of Three (Fragment Ro3)", fmt_rule(&d.ro3_fragment)),
                    ("Beyond Ro5 (bRo5)", yes_no(d.beyond_ro5).to_string()),
                    (
                        "Bioavailability Score (Abbott)",
                        format!("{:.2}", d.bioavailability_score),
                    ),
                ],
            );
        });

    egui::CollapsingHeader::new("Medicinal Chemistry & Safety Alerts")
        .default_open(true)
        .show(ui, |ui| {
            let pains_txt = if mc.pains_alerts.is_empty() {
                format!(
                    "0 alerts ({} curated patterns checked)",
                    mc.pains_catalog_size
                )
            } else {
                format!(
                    "{}/{} alert(s): {}",
                    mc.pains_alerts.len(),
                    mc.pains_catalog_size,
                    mc.pains_alerts.join(", ")
                )
            };
            let brenk_txt = if mc.brenk_alerts.is_empty() {
                format!(
                    "0 alerts ({} curated patterns checked)",
                    mc.brenk_catalog_size
                )
            } else {
                format!(
                    "{}/{} alert(s): {}",
                    mc.brenk_alerts.len(),
                    mc.brenk_catalog_size,
                    mc.brenk_alerts.join(", ")
                )
            };
            let herg_txt = if s.herg_reasons.is_empty() {
                s.herg_risk.as_str().to_string()
            } else {
                format!("{} ({})", s.herg_risk.as_str(), s.herg_reasons.join("; "))
            };
            let ames_txt = if s.ames_alerts.is_empty() {
                "0 alerts".to_string()
            } else {
                format!("{}: {}", s.ames_alerts.len(), s.ames_alerts.join(", "))
            };
            let dili_txt = if s.dili_alerts.is_empty() {
                "0 alerts".to_string()
            } else {
                format!("{}: {}", s.dili_alerts.len(), s.dili_alerts.join(", "))
            };
            kv_grid(
                ui,
                "mc_grid",
                &[
                    ("PAINS", pains_txt),
                    ("Brenk", brenk_txt),
                    ("Leadlikeness", fmt_rule(&mc.leadlikeness)),
                    (
                        "SA score (heuristic proxy)",
                        format!("{:.2}", mc.synthetic_accessibility),
                    ),
                    ("hERG Risk (heur.)", herg_txt),
                    ("AMES Mutagenicity (heur.)", ames_txt),
                    ("DILI Reactive Met. (heur.)", dili_txt),
                ],
            );
        });
}

fn kv_grid(ui: &mut egui::Ui, id: &str, rows: &[(&str, String)]) {
    egui::Grid::new(id)
        .striped(true)
        .spacing([20.0, 4.0])
        .show(ui, |ui| {
            for (k, v) in rows {
                ui.label(*k);
                ui.strong(v);
                ui.end_row();
            }
        });
}

fn fmt_rule(r: &admet_calc::RuleResult) -> String {
    if r.details.is_empty() {
        format!("{}; {} violations", yes_no(r.pass), r.violations)
    } else {
        format!(
            "{}; {} violation(s): {}",
            yes_no(r.pass),
            r.violations,
            r.details.join(", ")
        )
    }
}

fn yes_no(b: bool) -> &'static str {
    if b { "Yes" } else { "No" }
}

fn main() -> Result<(), eframe::Error> {
    let startup_file = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .filter(|p| p.is_file());
    let title = format!(
        "AdmetCraft v{} — Local-First ADMET Workbench",
        env!("CARGO_PKG_VERSION")
    );
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(title)
            .with_inner_size([1280.0, 820.0]),
        ..Default::default()
    };
    eframe::run_native(
        "AdmetCraft",
        options,
        Box::new(move |_cc| Ok(Box::new(AdmetCraftApp::from_startup_path(startup_file)))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sketcher_loads_and_syncs_smiles() -> Result<(), chem_core::ChemError> {
        let mut sk = MoleculeSketcher::default();
        sk.load_smiles("CC(=O)Oc1ccccc1C(=O)O")?;
        assert!(!sk.cached_smiles.is_empty());
        assert_eq!(sk.mol.heavy_atom_count(), 13);
        Ok(())
    }
}
