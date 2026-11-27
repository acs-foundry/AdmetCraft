//! Floating, Movable, Resizable & Expandable Mini-Application Window Manager for AdmetCraft.
//!
//! Organizes every AdmetCraft capability as a self-contained floating mini-application window
//! with a crisp Spanish explanatory header (*Qué hace*, *Cuándo usarla*, *Flujo I/O*),
//! drag-to-move, free resizing, and 1-click Expand/Restore (`🗖 Ampliar ventana` / `🗗 Tamaño normal`).

use std::collections::{BTreeMap, BTreeSet};

use eframe::egui;

/// Every standalone floating mini-application inside AdmetCraft.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MiniAppId {
    Tutorials,
    SketcherSmiles,
    AdmePhysChem,
    ToxicityQsar,
    PkPbpkDose,
    BatchScreeningExport,
}

/// Descriptive metadata and initial geometry for one floating mini-application.
#[derive(Debug, Clone, Copy)]
pub struct MiniAppMeta {
    pub id: MiniAppId,
    pub short_label: &'static str,
    pub window_title: &'static str,
    pub what_it_does: &'static str,
    pub when_to_use: &'static str,
    pub io_summary: &'static str,
    pub default_size: [f32; 2],
    pub default_offset: [f32; 2],
    pub accent_rgb: [u8; 3],
}

/// All floating mini-applications displayed in the top launcher bar.
pub const ALL_MINI_APPS: &[MiniAppMeta] = &[
    MiniAppMeta {
        id: MiniAppId::Tutorials,
        short_label: "📚 Tutoriales",
        window_title: "📚 Tutoriales Interactivos · Aprende AdmetCraft Paso a Paso",
        what_it_does: "Guías interactivas paso a paso en español con ejecución real en 1 clic sobre fármacos, alertas, PBPK y bibliotecas.",
        when_to_use: "Ideal al empezar o para dominar los flujos de ADME, BOILED-Egg, PAINS/hERG, dosis clínica PBPK y cribado batch.",
        io_summary: "Guía viva → ejecuta acciones reales en AdmetCraft y abre la mini-app correspondiente.",
        default_size: [540.0, 560.0],
        default_offset: [48.0, 96.0],
        accent_rgb: [16, 185, 129],
    },
    MiniAppMeta {
        id: MiniAppId::SketcherSmiles,
        short_label: "✏ Editor 2D & SMILES",
        window_title: "✏ Editor Molecular 2D, Lista SMILES e Importación (.smi/.csv/.sdf)",
        what_it_does: "Dibuja o edita estructuras 2D interactivamente, sincroniza SMILES canónicos y carga lotes `.smi`, `.csv` o `.sdf` en sandbox.",
        when_to_use: "Para modificar un sustituyente en vivo, pegar SMILES o importar candidatos desde ChemCraft y DockCraft.",
        io_summary: "Dibujo 2D / SMILES / archivo `.smi|.csv|.sdf` → sesión evaluada ADMET + alertas + QSAR.",
        default_size: [520.0, 620.0],
        default_offset: [92.0, 104.0],
        accent_rgb: [6, 182, 212],
    },
    MiniAppMeta {
        id: MiniAppId::AdmePhysChem,
        short_label: "🧪 ADME, Reglas & Egg",
        window_title: "🧪 Perfil ADME, Fisicoquímica, Radar, BOILED-Egg y Reglas (Lipinski/CNS MPO/QED)",
        what_it_does: "Inspecciona el Radar de Biodisponibilidad, LogP/LogD7.4, pKa, solubilidad ESOL/Ali, GI/BBB/P-gp, CYP, reglas industriales y BOILED-Egg interactivo.",
        when_to_use: "Para perfilar propiedades fisicoquímicas, permeabilidad oral/SNC y drug-likeness (Lipinski, Veber, QED, CNS MPO, bRo5).",
        io_summary: "Molécula activa / lote → ficha ADME completa, mapa 2D con SoM, Radar y BOILED-Egg clicable.",
        default_size: [650.0, 640.0],
        default_offset: [136.0, 112.0],
        accent_rgb: [59, 130, 246],
    },
    MiniAppMeta {
        id: MiniAppId::ToxicityQsar,
        short_label: "☣ Toxicidad & QSAR",
        window_title: "☣ Toxicidad, Alertas (PAINS/Brenk/hERG/DILI), QSAR TDC, SoM, Tox21 y Metabolite Guard",
        what_it_does: "Detecta PAINS, Brenk, AMES, DILI, riesgo hERG, vías Tox21 (12 ensayos), off-targets SEA Bowes-20, Metabolite Guard, QSAR TDC (IC 90% + AD) y sitios SoM.",
        when_to_use: "Antes de avanzar un hit a síntesis o ensayo: descarta falsos positivos, cardiotoxicidad hERG, bioactivación reactiva y puntos lábiles CYP.",
        io_summary: "Compuesto seleccionado → alertas estructurales, tabla QSAR con IC 90%, hotspots 2D SoM y metabolitos.",
        default_size: [680.0, 650.0],
        default_offset: [180.0, 118.0],
        accent_rgb: [239, 68, 68],
    },
    MiniAppMeta {
        id: MiniAppId::PkPbpkDose,
        short_label: "💊 PK / PBPK y Dosis",
        window_title: "💊 Simulador Humano PK / Minimal-PBPK y Dosis Clínica (QD / BID)",
        what_it_does: "Predice dosis terapéutica humana (QD/BID + IC 90%), biodisponibilidad oral F (Fa·Fg·Fh), clase BCS, CL/Vd/t½, margen hERG y curva C_u(t) 48 h.",
        when_to_use: "Cuando conoces la potencia bioquímica/celular (IC50/Ki) y necesitas estimar viabilidad clínica oral, IV o SC.",
        io_summary: "Molécula + IC50/Ki (nM) + ruta (Oral/IV/SC) → pauta recomendada, curva C_u(t) y alertas traslacionales.",
        default_size: [620.0, 620.0],
        default_offset: [224.0, 124.0],
        accent_rgb: [139, 92, 246],
    },
    MiniAppMeta {
        id: MiniAppId::BatchScreeningExport,
        short_label: "📊 Lote, MPO & Export",
        window_title: "📊 Cribado de Bibliotecas (CSV/SDF), Ranking TPP MPO y Exportación Segura",
        what_it_does: "Filtra bibliotecas multiparamétricamente, prioriza hits con Target Product Profile (TPP MPO), ordena la tabla completa y exporta CSV/JSON/PDF/SVG.",
        when_to_use: "Para cribar decenas o miles de compuestos desde ChemCraft/DockCraft, comparar TPP (Oral/CNS/Oncology) y generar informes auditables.",
        io_summary: "Biblioteca batch → embudo ADMET + ranking TPP MPO → informes CSV, JSON, PDF y BOILED-Egg SVG.",
        default_size: [700.0, 640.0],
        default_offset: [268.0, 130.0],
        accent_rgb: [245, 158, 11],
    },
];

/// Return metadata for a given `MiniAppId`.
#[must_use]
pub fn mini_app_meta(id: MiniAppId) -> &'static MiniAppMeta {
    ALL_MINI_APPS
        .iter()
        .find(|m| m.id == id)
        .unwrap_or(&ALL_MINI_APPS[0])
}

/// Active sub-tab selection for compound mini-apps that unite related tools cleanly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompoundSubTab {
    Primary,
    Secondary,
}

/// Sorting mode for the Batch Screening & TPP MPO ranking table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BatchSortMode {
    InputOrder,
    #[default]
    TppMpoDesc,
    QedDesc,
    CnsMpoDesc,
}

impl BatchSortMode {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::InputOrder => "Orden original",
            Self::TppMpoDesc => "Ranking TPP MPO ↓",
            Self::QedDesc => "Ranking QED ↓",
            Self::CnsMpoDesc => "Ranking CNS MPO ↓",
        }
    }
}

/// Manages open/expanded floating mini-app windows and detachable workspace panels.
#[derive(Debug, Clone)]
pub struct WorkspaceWindows {
    pub open_apps: BTreeSet<MiniAppId>,
    pub expanded_apps: BTreeSet<MiniAppId>,
    pub sub_tabs: BTreeMap<MiniAppId, CompoundSubTab>,
    /// Allow detaching the left 2D Sketcher & SMILES Input panel as a floating movable window.
    pub left_panel_floating: bool,
    pub show_left_panel: bool,
}

impl Default for WorkspaceWindows {
    fn default() -> Self {
        let mut open_apps = BTreeSet::new();
        // Open Interactive Tutorials by default so new users can walk through every workflow immediately.
        open_apps.insert(MiniAppId::Tutorials);
        Self {
            open_apps,
            expanded_apps: BTreeSet::new(),
            sub_tabs: BTreeMap::new(),
            left_panel_floating: false,
            show_left_panel: true,
        }
    }
}

impl WorkspaceWindows {
    pub fn open(&mut self, id: MiniAppId) {
        self.open_apps.insert(id);
    }

    pub fn close(&mut self, id: MiniAppId) {
        self.open_apps.remove(&id);
        self.expanded_apps.remove(&id);
    }

    pub fn close_all(&mut self) {
        self.open_apps.clear();
        self.expanded_apps.clear();
    }

    pub fn toggle(&mut self, id: MiniAppId) {
        if !self.open_apps.remove(&id) {
            self.open_apps.insert(id);
        } else {
            self.expanded_apps.remove(&id);
        }
    }

    pub fn toggle_expand(&mut self, id: MiniAppId) {
        if !self.expanded_apps.remove(&id) {
            self.expanded_apps.insert(id);
        }
    }

    #[must_use]
    pub fn is_open(&self, id: MiniAppId) -> bool {
        self.open_apps.contains(&id)
    }

    #[must_use]
    pub fn is_expanded(&self, id: MiniAppId) -> bool {
        self.expanded_apps.contains(&id)
    }

    #[must_use]
    pub fn sub_tab(&self, id: MiniAppId) -> CompoundSubTab {
        self.sub_tabs
            .get(&id)
            .copied()
            .unwrap_or(CompoundSubTab::Primary)
    }

    pub fn set_sub_tab(&mut self, id: MiniAppId, tab: CompoundSubTab) {
        self.sub_tabs.insert(id, tab);
    }
}

/// Compute a safe initial window position clamped inside `screen` (`ctx.screen_rect()`).
#[must_use]
pub fn clamped_window_pos(screen: egui::Rect, offset: [f32; 2]) -> egui::Pos2 {
    let min_x = screen.left() + 24.0;
    let max_x = (screen.right() - 420.0).max(min_x);
    let min_y = screen.top() + 88.0;
    let max_y = (screen.bottom() - 340.0).max(min_y);
    let x = (screen.left() + offset[0]).clamp(min_x, max_x);
    let y = (screen.top() + offset[1]).clamp(min_y, max_y);
    egui::pos2(x, y)
}

/// Compute the expanded floating window size relative to `screen` (`ctx.screen_rect()`).
#[must_use]
pub fn expanded_window_size(screen: egui::Rect) -> egui::Vec2 {
    egui::vec2(
        (screen.width() * 0.84).max(640.0),
        (screen.height() * 0.80).max(520.0),
    )
}

/// Paint a crisp, plain-Spanish educational banner at the top of every floating mini-application.
pub fn render_mini_app_header(ui: &mut egui::Ui, meta: &MiniAppMeta, is_expanded: bool) -> bool {
    let mut toggle_expand_clicked = false;
    let [r, g, b] = meta.accent_rgb;
    let accent = egui::Color32::from_rgb(r, g, b);
    let bg = egui::Color32::from_rgba_unmultiplied(r, g, b, 24);

    egui::Frame::group(ui.style())
        .fill(bg)
        .stroke(egui::Stroke::new(1.2_f32, accent))
        .corner_radius(egui::CornerRadius::same(8))
        .inner_margin(egui::Margin::symmetric(10, 7))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.colored_label(
                    accent,
                    egui::RichText::new(meta.short_label).strong().size(13.5),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let lbl = if is_expanded {
                        "🗗 Tamaño normal"
                    } else {
                        "🗖 Ampliar ventana"
                    };
                    if ui
                        .small_button(lbl)
                        .on_hover_text(
                            "Alterna entre tamaño normal y ventana ampliada (también puedes arrastrar y estirar cualquier borde)",
                        )
                        .clicked()
                    {
                        toggle_expand_clicked = true;
                    }
                });
            });
            ui.label(egui::RichText::new(meta.what_it_does).size(12.0));
            ui.small(format!("Cuándo usarla: {}", meta.when_to_use));
            ui.small(format!("Flujo I/O: {}", meta.io_summary));
        });
    ui.add_space(4.0);
    toggle_expand_clicked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_windows_tracks_open_expand_and_subtab_state() {
        let mut ws = WorkspaceWindows::default();
        assert!(ws.is_open(MiniAppId::Tutorials));
        assert!(ws.show_left_panel);
        assert!(!ws.left_panel_floating);
        assert_eq!(ws.sub_tab(MiniAppId::AdmePhysChem), CompoundSubTab::Primary);

        ws.open(MiniAppId::ToxicityQsar);
        assert!(ws.is_open(MiniAppId::ToxicityQsar));
        ws.set_sub_tab(MiniAppId::ToxicityQsar, CompoundSubTab::Secondary);
        assert_eq!(
            ws.sub_tab(MiniAppId::ToxicityQsar),
            CompoundSubTab::Secondary
        );

        ws.toggle_expand(MiniAppId::ToxicityQsar);
        assert!(ws.is_expanded(MiniAppId::ToxicityQsar));
        ws.toggle(MiniAppId::ToxicityQsar);
        assert!(!ws.is_open(MiniAppId::ToxicityQsar));
        assert!(!ws.is_expanded(MiniAppId::ToxicityQsar));

        ws.open(MiniAppId::PkPbpkDose);
        ws.close(MiniAppId::PkPbpkDose);
        assert!(!ws.is_open(MiniAppId::PkPbpkDose));

        ws.open(MiniAppId::BatchScreeningExport);
        ws.close_all();
        assert!(ws.open_apps.is_empty());
        assert!(ws.expanded_apps.is_empty());
    }

    #[test]
    fn all_mini_apps_metadata_is_complete_and_unique() {
        assert_eq!(ALL_MINI_APPS.len(), 6);
        let mut seen = BTreeSet::new();
        for meta in ALL_MINI_APPS {
            assert!(seen.insert(meta.id), "duplicate MiniAppId: {:?}", meta.id);
            assert!(!meta.short_label.is_empty());
            assert!(!meta.window_title.is_empty());
            assert!(!meta.what_it_does.is_empty());
            assert!(!meta.when_to_use.is_empty());
            assert!(!meta.io_summary.is_empty());
            assert!(meta.default_size[0] >= 400.0 && meta.default_size[1] >= 400.0);
            assert_eq!(mini_app_meta(meta.id).id, meta.id);
        }
    }

    #[test]
    fn clamped_window_pos_and_expanded_size_stay_within_screen_bounds() {
        let standard = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1280.0, 820.0));
        let pos = clamped_window_pos(standard, [2000.0, 2000.0]);
        assert!(pos.x <= standard.right() - 420.0);
        assert!(pos.y <= standard.bottom() - 340.0);
        assert!(pos.x >= standard.left() + 24.0);
        assert!(pos.y >= standard.top() + 88.0);

        let tiny = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(300.0, 200.0));
        let tiny_pos = clamped_window_pos(tiny, [500.0, 500.0]);
        assert_eq!(tiny_pos, egui::pos2(24.0, 88.0));

        let exp = expanded_window_size(standard);
        assert!(exp.x >= 640.0 && exp.y >= 520.0);
    }
}
