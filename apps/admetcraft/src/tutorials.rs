//! Interactive In-App Spanish Guided Tutorials for AdmetCraft.
//!
//! Each tutorial walks the user step-by-step through a real ADMET, toxicity, PBPK, or batch
//! screening workflow inside the live workbench, explaining *what* the tool evaluates, *what to
//! observe*, and providing a 1-click `▶ Ejecutar este paso ahora` action that runs real
//! `AdmetCraftApp` commands and opens the matching floating mini-application.

use std::collections::BTreeSet;

use admet_calc::{PkRoute, TppPreset};
use admet_engine::FilterCriteria;

use crate::miniapps::{BatchSortMode, CompoundSubTab, MiniAppId};

/// Curated 3-drug comparison batch including Imatinib, Ibuprofen, and Atorvastatin.
pub const TUTORIAL_REFERENCE_DRUGS_BATCH: &str = "\
Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1 Imatinib
CC(C)Cc1ccc(C(C)C(=O)O)cc1 Ibuprofen
CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O Atorvastatin";

/// Curated MedChem safety optimization series: PAINS/DILI false positive + high-hERG hit vs clean lead.
pub const TUTORIAL_TOX_OPTIMIZATION_BATCH: &str = "\
S=C1NC(=O)C(=Cc2ccc(O)c(O)c2)S1 Hit_PAINS_Rhodanine
CC/C(=C(/c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1 Hit_hERG_Tamoxifen
COc1ccc(-c2cc(C(=O)N3CCOCC3)ncc2)cc1 Lead_Optimizado";

/// Curated 8-compound virtual screening library simulating a ChemCraft / DockCraft handoff.
pub const TUTORIAL_CHEMCRAFT_DOCKCRAFT_BATCH: &str = "\
COc1ccc(-c2cc(C(=O)N3CCOCC3)ncc2)cc1 Hit_01_Lead_Morpholine
CC(C)Cc1ccc(C(C)C(=O)O)cc1 Hit_02_Ibuprofen_Core
CN1c2ccc(Cl)cc2C(=NCC1=O)c1ccccc1 Hit_03_Diazepam_CNS
Cc1ccc(C(=O)Nc2ccc(F)cn2)cc1 Hit_04_Amide_Clean
S=C1NC(=O)C(=Cc2ccc(O)c(O)c2)S1 Hit_05_PAINS_Rhodanine
O=C1C=CC(=O)c2ccccc12 Hit_06_PAINS_Quinone
CC/C(=C(/c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1 Hit_07_High_hERG
CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O Hit_08_bRo5_Large";

/// Concrete action triggered when the user clicks `▶ Ejecutar este paso ahora` in a tutorial step.
#[derive(Debug, Clone)]
pub enum TutorialAction {
    /// Load the 15 built-in reference drugs, select one by name, and open a floating mini-app.
    LoadExamplesAndSelect {
        molecule_name: &'static str,
        open_app: MiniAppId,
        sub_tab: CompoundSubTab,
    },
    /// Select a molecule in the active session (loading examples if absent) and open a mini-app.
    SelectMoleculeByName {
        molecule_name: &'static str,
        open_app: MiniAppId,
        sub_tab: CompoundSubTab,
    },
    /// Evaluate a custom SMILES batch, select a target index, optionally set TPP/sort, and open a mini-app.
    EvaluateCustomBatch {
        input: &'static str,
        select_index: usize,
        tpp_preset: Option<TppPreset>,
        sort_mode: Option<BatchSortMode>,
        open_app: MiniAppId,
        sub_tab: CompoundSubTab,
    },
    /// Switch Target Product Profile (TPP) preset, recompute MPO scores, and focus a molecule.
    SetTppAndSelect {
        preset: TppPreset,
        molecule_name: &'static str,
        open_app: MiniAppId,
        sub_tab: CompoundSubTab,
    },
    /// Apply a real multi-parameter `FilterCriteria` to the loaded session batch.
    ApplyFilter {
        criteria: FilterCriteria,
        open_app: MiniAppId,
        sub_tab: CompoundSubTab,
    },
    /// Configure and execute the live Human PK / Minimal-PBPK dose simulator for a target molecule.
    RunPkSimulation {
        molecule_name: &'static str,
        potency_nm: f64,
        route: PkRoute,
        open_app: MiniAppId,
    },
    /// Open a specific floating mini-application on a given sub-tab.
    OpenMiniApp {
        app: MiniAppId,
        sub_tab: CompoundSubTab,
    },
}

/// One guided step inside an interactive tutorial track.
#[derive(Debug, Clone)]
pub struct TutorialStep {
    pub title: &'static str,
    pub objective: &'static str,
    pub what_to_look_at: &'static str,
    pub action_label: &'static str,
    pub action: TutorialAction,
    pub related_app: Option<MiniAppId>,
}

/// Complete multi-step interactive tutorial track.
#[derive(Debug, Clone)]
pub struct TutorialTrack {
    pub id: &'static str,
    pub title: &'static str,
    pub duration_min: u8,
    pub summary: &'static str,
    pub steps: Vec<TutorialStep>,
}

/// Runtime state of the Interactive Tutorials mini-app.
#[derive(Debug, Clone)]
pub struct TutorialState {
    pub active_track_idx: usize,
    pub active_step_idx: usize,
    pub completed_steps: BTreeSet<(usize, usize)>,
    pub last_feedback: String,
}

impl Default for TutorialState {
    fn default() -> Self {
        Self {
            active_track_idx: 0,
            active_step_idx: 0,
            completed_steps: BTreeSet::new(),
            last_feedback: "Selecciona un tutorial y pulsa '▶ Ejecutar este paso ahora' para practicar sobre moléculas reales.".to_string(),
        }
    }
}

impl TutorialState {
    pub fn mark_completed(&mut self, track_idx: usize, step_idx: usize, feedback: String) {
        self.completed_steps.insert((track_idx, step_idx));
        self.last_feedback = feedback;
    }

    #[must_use]
    pub fn completed_in_track(&self, track_idx: usize, total_steps: usize) -> usize {
        (0..total_steps)
            .filter(|&s| self.completed_steps.contains(&(track_idx, s)))
            .count()
    }
}

/// Build the 5 curated interactive Spanish tutorials covering every AdmetCraft workflow.
#[must_use]
pub fn built_in_tutorials() -> Vec<TutorialTrack> {
    vec![
        TutorialTrack {
            id: "reference_drugs_adme",
            title: "1. Fármacos de Referencia: ADME, Radar y Editor 2D",
            duration_min: 3,
            summary: "Evalúa Ibuprofeno, Atorvastatina e Imatinib para comparar Radar de Biodisponibilidad, ionización LogD7.4, reglas Lipinski/QED/bRo5 y edición 2D en vivo.",
            steps: vec![
                TutorialStep {
                    title: "Evaluar Ibuprofeno: perfil oral óptimo y estado ácido",
                    objective: "Carga los 15 fármacos clínicos de referencia, enfoca Ibuprofeno y abre la mini-app flotante 'ADME, Reglas & Egg' con su Radar de Biodisponibilidad.",
                    what_to_look_at: "Comprueba cómo los 6 ejes del Radar encajan en la zona rosa óptima, el estado ácido a pH 7.4 (LogD7.4 mucho menor que cLogP), 0 violaciones Lipinski y alto QED.",
                    action_label: "▶ Ejecutar este paso ahora (Cargar Ibuprofeno)",
                    action: TutorialAction::LoadExamplesAndSelect {
                        molecule_name: "Ibuprofen",
                        open_app: MiniAppId::AdmePhysChem,
                        sub_tab: CompoundSubTab::Primary,
                    },
                    related_app: Some(MiniAppId::AdmePhysChem),
                },
                TutorialStep {
                    title: "Contrastar con Atorvastatina (fármaco Beyond Rule of 5)",
                    objective: "Cambia el foco a Atorvastatina dentro del mismo lote para observar cómo AdmetCraft diagnostica moléculas de alto peso molecular (bRo5).",
                    what_to_look_at: "Observa cómo sobresalen SIZE y LIPO en el Radar, `Beyond Ro5 (bRo5)` marca `Yes`, cambia la solubilidad ESOL y aparecen violaciones en Lipinski/Ghose.",
                    action_label: "▶ Ejecutar este paso ahora (Inspeccionar Atorvastatina)",
                    action: TutorialAction::SelectMoleculeByName {
                        molecule_name: "Atorvastatin",
                        open_app: MiniAppId::AdmePhysChem,
                        sub_tab: CompoundSubTab::Primary,
                    },
                    related_app: Some(MiniAppId::AdmePhysChem),
                },
                TutorialStep {
                    title: "Cargar Imatinib en el Editor 2D & SMILES interactivo",
                    objective: "Evalúa Imatinib junto con Ibuprofeno y Atorvastatina, y sincronízalo en el editor 2D para probar modificaciones químicas en tiempo real.",
                    what_to_look_at: "En la ventana flotante 'Editor 2D & SMILES' verás la estructura 2D de Imatinib y su SMILES canónico listos para añadir anillos o heteroátomos y re-evaluar en 1 clic.",
                    action_label: "▶ Ejecutar este paso ahora (Abrir Imatinib en Editor 2D)",
                    action: TutorialAction::EvaluateCustomBatch {
                        input: TUTORIAL_REFERENCE_DRUGS_BATCH,
                        select_index: 0,
                        tpp_preset: Some(TppPreset::OncologyKinase),
                        sort_mode: None,
                        open_app: MiniAppId::SketcherSmiles,
                        sub_tab: CompoundSubTab::Primary,
                    },
                    related_app: Some(MiniAppId::SketcherSmiles),
                },
            ],
        },
        TutorialTrack {
            id: "cns_mpo_boiled_egg",
            title: "2. Penetración SNC: CNS MPO y Gráfico BOILED-Egg",
            duration_min: 4,
            summary: "Distingue fármacos penetrantes cerebrales (yema BBB+) de compuestos periféricos (clara HIA), analiza P-gp y filtra candidatos óptimos para el SNC.",
            steps: vec![
                TutorialStep {
                    title: "Explorar los 15 fármacos en el gráfico interactivo BOILED-Egg",
                    objective: "Carga los 15 fármacos de referencia, selecciona Diazepam y abre la pestaña BOILED-Egg (TPSA vs WLOGP) dentro de la mini-app ADME.",
                    what_to_look_at: "La yema amarilla marca penetración cerebral (BBB+) y la clara blanca absorción intestinal (HIA). Puntos azules = PGP+, rojos = PGP-. Pasa el cursor por cada punto para ver su 2D.",
                    action_label: "▶ Ejecutar este paso ahora (Ver BOILED-Egg con Diazepam)",
                    action: TutorialAction::LoadExamplesAndSelect {
                        molecule_name: "Diazepam",
                        open_app: MiniAppId::AdmePhysChem,
                        sub_tab: CompoundSubTab::Secondary,
                    },
                    related_app: Some(MiniAppId::AdmePhysChem),
                },
                TutorialStep {
                    title: "Activar el Target Product Profile (TPP) 'CNS Penetrant'",
                    objective: "Recalcula la puntuación TPP MPO de todo el lote bajo criterios cerebrales (CNS MPO ≥ 4, BBB+, baja salida P-gp) y audita el scorecard de Diazepam.",
                    what_to_look_at: "Revisa la tabla de objetivos TPP con semáforo PASS/WARN/FAIL y la puntuación Pfizer CNS MPO (sobre 6.0) en la ventana 'Lote, MPO & Export'.",
                    action_label: "▶ Ejecutar este paso ahora (Aplicar TPP CNS Penetrant)",
                    action: TutorialAction::SetTppAndSelect {
                        preset: TppPreset::CnsPenetrant,
                        molecule_name: "Diazepam",
                        open_app: MiniAppId::BatchScreeningExport,
                        sub_tab: CompoundSubTab::Primary,
                    },
                    related_app: Some(MiniAppId::BatchScreeningExport),
                },
                TutorialStep {
                    title: "Filtrar el lote a candidatos CNS MPO ≥ 4.0 y BBB+",
                    objective: "Aplica el filtro combinado de SNC directamente sobre el lote activo (reversible con Undo / Ctrl+Z) y observa el resultado en BOILED-Egg.",
                    what_to_look_at: "Solo sobreviven los fármacos situados en la yema amarilla con CNS MPO ≥ 4.0. Pulsa 'Undo' en la barra superior cuando quieras restaurar los 15 compuestos.",
                    action_label: "▶ Ejecutar este paso ahora (Filtrar CNS MPO ≥ 4 y BBB+)",
                    action: TutorialAction::ApplyFilter {
                        criteria: FilterCriteria {
                            min_cns_mpo: Some(4.0),
                            require_bbb: Some(true),
                            ..FilterCriteria::default()
                        },
                        open_app: MiniAppId::AdmePhysChem,
                        sub_tab: CompoundSubTab::Secondary,
                    },
                    related_app: Some(MiniAppId::AdmePhysChem),
                },
            ],
        },
        TutorialTrack {
            id: "toxicity_alerts_lead_opt",
            title: "3. Alertas PAINS / hERG / DILI y Optimización de Lead",
            duration_min: 4,
            summary: "Identifica falsos positivos PAINS, toxicóforos Brenk/DILI, cardiotoxicidad hERG y sitios metabólicos CYP450, y valida cómo un lead optimizado elimina los riesgos.",
            steps: vec![
                TutorialStep {
                    title: "Cazar alertas PAINS, Brenk y DILI en un hit problemático",
                    objective: "Evalúa una mini-serie química con un hit rodanina-catecol (PAINS/DILI), un hit básico lipófilo (hERG alto) y un lead optimizado limpio.",
                    what_to_look_at: "En 'Toxicidad & QSAR' observa las alertas PAINS (`ene_rhodanine_A`, `catechol_A`), Brenk y DILI (bioactivación o-quinona/rodanina) marcadas sobre `Hit_PAINS_Rhodanine`.",
                    action_label: "▶ Ejecutar este paso ahora (Auditar Hit PAINS/DILI)",
                    action: TutorialAction::EvaluateCustomBatch {
                        input: TUTORIAL_TOX_OPTIMIZATION_BATCH,
                        select_index: 0,
                        tpp_preset: Some(TppPreset::OralSmallMolecule),
                        sort_mode: None,
                        open_app: MiniAppId::ToxicityQsar,
                        sub_tab: CompoundSubTab::Primary,
                    },
                    related_app: Some(MiniAppId::ToxicityQsar),
                },
                TutorialStep {
                    title: "Inspeccionar riesgo hERG Alto, QSAR TDC y Sitios SoM",
                    objective: "Selecciona `Hit_hERG_Tamoxifen` y abre la vista de QSAR Cuantitativo (IC 90% + AD) y mapa 2D de Sitios de Metabolismo (CYP450/Fase II).",
                    what_to_look_at: "Verás el diagnóstico hERG High (amina básica + alta lipofilicidad + anillos aromáticos), los hotspots metabólicos en 2D con sugerencias `MedChem Fix` y los ensayos QSAR/Tox21/SEA.",
                    action_label: "▶ Ejecutar este paso ahora (Ver hERG Alto + Mapa SoM)",
                    action: TutorialAction::SelectMoleculeByName {
                        molecule_name: "Hit_hERG_Tamoxifen",
                        open_app: MiniAppId::ToxicityQsar,
                        sub_tab: CompoundSubTab::Secondary,
                    },
                    related_app: Some(MiniAppId::ToxicityQsar),
                },
                TutorialStep {
                    title: "Validar el Lead Optimizado limpio (0 PAINS, 0 DILI, hERG Bajo)",
                    objective: "Enfoca `Lead_Optimizado` (morfolina-amida neutra y polar) para confirmar la desaparición de las alertas estructurales y la caída del riesgo cardiotóxico.",
                    what_to_look_at: "Verifica 0 alertas PAINS, 0 Brenk, 0 AMES, 0 DILI y `hERG: Low`, demostrando el valor de iterar estructura → ADMET antes de sintetizar.",
                    action_label: "▶ Ejecutar este paso ahora (Validar Lead Optimizado)",
                    action: TutorialAction::SelectMoleculeByName {
                        molecule_name: "Lead_Optimizado",
                        open_app: MiniAppId::ToxicityQsar,
                        sub_tab: CompoundSubTab::Primary,
                    },
                    related_app: Some(MiniAppId::ToxicityQsar),
                },
            ],
        },
        TutorialTrack {
            id: "pk_pbpk_clinical_dose",
            title: "4. Simulación Humana PK / Minimal-PBPK y Dosis Clínica",
            duration_min: 3,
            summary: "Traduce potencia bioquímica (IC50/Ki) y propiedades ADMET en dosis humanas estimadas (QD/BID), biodisponibilidad F, clase BCS y perfiles C_u(t) en estado estacionario.",
            steps: vec![
                TutorialStep {
                    title: "Simular pauta oral QD/BID de Ibuprofeno (IC50 = 10 nM)",
                    objective: "Abre la mini-app 'PK / PBPK y Dosis' sobre Ibuprofeno en ruta Oral con una potencia objetivo de 10 nM.",
                    what_to_look_at: "Revisa la pauta recomendada, F = Fa × Fg × Fh, CL_tot, V_d,ss, t½, margen hERG y la curva azul de concentración plasmática libre C_u(t) frente a la línea verde IC50.",
                    action_label: "▶ Ejecutar este paso ahora (Simular Ibuprofeno Oral 10 nM)",
                    action: TutorialAction::RunPkSimulation {
                        molecule_name: "Ibuprofen",
                        potency_nm: 10.0,
                        route: PkRoute::Oral,
                        open_app: MiniAppId::PkPbpkDose,
                    },
                    related_app: Some(MiniAppId::PkPbpkDose),
                },
                TutorialStep {
                    title: "Evaluar impacto de baja potencia y solubilidad en Atorvastatina",
                    objective: "Cambia a Atorvastatina con IC50 = 150 nM (Oral) para ver cómo la solubilidad BCS y menor potencia disparan la dosis diaria y las advertencias.",
                    what_to_look_at: "Compara la dosis QD/BID (mg/d y su intervalo 90%), la clase BCS y las advertencias de exposición o margen de seguridad en la parte inferior.",
                    action_label: "▶ Ejecutar este paso ahora (Simular Atorvastatina 150 nM)",
                    action: TutorialAction::RunPkSimulation {
                        molecule_name: "Atorvastatin",
                        potency_nm: 150.0,
                        route: PkRoute::Oral,
                        open_app: MiniAppId::PkPbpkDose,
                    },
                    related_app: Some(MiniAppId::PkPbpkDose),
                },
                TutorialStep {
                    title: "Comparar administración Intravenosa (IV) en Ciprofloxacino",
                    objective: "Cambia la ruta a Intravenosa (IV) sobre Ciprofloxacino con IC50 = 25 nM para contrastar el perfil plasmático sin efecto de primer paso.",
                    what_to_look_at: "Fíjate en los picos inmediatos Cmax,u de la curva C_u(t) tras cada administración IV y en la caída exponencial controlada por CL_tot y V_d,ss.",
                    action_label: "▶ Ejecutar este paso ahora (Simular Ciprofloxacino IV 25 nM)",
                    action: TutorialAction::RunPkSimulation {
                        molecule_name: "Ciprofloxacin",
                        potency_nm: 25.0,
                        route: PkRoute::Iv,
                        open_app: MiniAppId::PkPbpkDose,
                    },
                    related_app: Some(MiniAppId::PkPbpkDose),
                },
            ],
        },
        TutorialTrack {
            id: "batch_library_screening",
            title: "5. Cribado de Bibliotecas (ChemCraft / DockCraft) y Exportación",
            duration_min: 4,
            summary: "Evalúa una mini-biblioteca virtual procedente de ChemCraft/DockCraft, aplica un embudo multiparamétrico (Lipinski, PAINS, QED, hERG), rankea por TPP MPO y exporta informes.",
            steps: vec![
                TutorialStep {
                    title: "Cargar mini-biblioteca virtual de 8 hits (ChemCraft / DockCraft)",
                    objective: "Evalua 8 candidatos mixtos (leads limpios, falsos positivos PAINS, un hit hERG alto y un compuesto bRo5) ordenados por Ranking TPP MPO.",
                    what_to_look_at: "En 'Lote, MPO & Export' observa la tabla ordenada de mayor a menor TPP MPO: los leads limpios suben a los primeros puestos mientras los PAINS/hERG caen al final.",
                    action_label: "▶ Ejecutar este paso ahora (Cargar biblioteca de 8 hits)",
                    action: TutorialAction::EvaluateCustomBatch {
                        input: TUTORIAL_CHEMCRAFT_DOCKCRAFT_BATCH,
                        select_index: 0,
                        tpp_preset: Some(TppPreset::OralSmallMolecule),
                        sort_mode: Some(BatchSortMode::TppMpoDesc),
                        open_app: MiniAppId::BatchScreeningExport,
                        sub_tab: CompoundSubTab::Primary,
                    },
                    related_app: Some(MiniAppId::BatchScreeningExport),
                },
                TutorialStep {
                    title: "Aplicar embudo de descarte (Lipinski 0 + Sin PAINS + QED ≥ 0.5 + hERG)",
                    objective: "Ejecuta un filtrado estricto en 1 clic para eliminar automáticamente alertas PAINS, riesgo hERG alto y violaciones de Lipinski.",
                    what_to_look_at: "La tabla se reduce de 8 compuestos a solo los 4 candidatos limpios listos para avanzar. Recuerda que puedes deshacer cualquier filtro con Undo (Ctrl+Z).",
                    action_label: "▶ Ejecutar este paso ahora (Filtrar biblioteca limpia)",
                    action: TutorialAction::ApplyFilter {
                        criteria: FilterCriteria {
                            max_lipinski_violations: Some(0),
                            no_pains: true,
                            min_qed: Some(0.50),
                            no_high_herg: true,
                            ..FilterCriteria::default()
                        },
                        open_app: MiniAppId::BatchScreeningExport,
                        sub_tab: CompoundSubTab::Primary,
                    },
                    related_app: Some(MiniAppId::BatchScreeningExport),
                },
                TutorialStep {
                    title: "Preparar importación SDF/CSV y exportación CSV / JSON / PDF / SVG",
                    objective: "Abre la sub-pestaña de Importación y Exportación Segura para revisar las métricas del embudo y los formatos de entrega reproducibles.",
                    what_to_look_at: "Desde aquí puedes importar archivos `.smi`/`.csv`/`.sdf` (o arrastrarlos a la ventana) y exportar en segundo plano CSV, JSON completo, informe PDF y gráfico BOILED-Egg SVG.",
                    action_label: "▶ Ejecutar este paso ahora (Abrir panel Import/Export)",
                    action: TutorialAction::OpenMiniApp {
                        app: MiniAppId::BatchScreeningExport,
                        sub_tab: CompoundSubTab::Secondary,
                    },
                    related_app: Some(MiniAppId::BatchScreeningExport),
                },
            ],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use admet_calc::HergRisk;

    #[test]
    fn built_in_tutorials_cover_five_tracks_with_non_empty_steps() {
        let tracks = built_in_tutorials();
        assert_eq!(tracks.len(), 5);
        for tr in &tracks {
            assert!(!tr.id.is_empty());
            assert!(!tr.title.is_empty());
            assert!(!tr.summary.is_empty());
            assert!(tr.steps.len() >= 2, "track {} should have >=2 steps", tr.id);
            for st in &tr.steps {
                assert!(!st.title.is_empty());
                assert!(!st.objective.is_empty());
                assert!(!st.what_to_look_at.is_empty());
                assert!(!st.action_label.is_empty());
            }
        }
        let mut state = TutorialState::default();
        assert_eq!(state.completed_in_track(0, tracks[0].steps.len()), 0);
        state.mark_completed(0, 0, "done".to_string());
        assert_eq!(state.completed_in_track(0, tracks[0].steps.len()), 1);
    }

    #[test]
    fn every_tutorial_step_executes_real_admetcraft_logic_without_parse_failures() {
        let mut app = crate::AdmetCraftApp::default();
        let tracks = built_in_tutorials();
        let mut total_steps = 0usize;

        for (t_idx, track) in tracks.iter().enumerate() {
            for (s_idx, step) in track.steps.iter().enumerate() {
                app.run_tutorial_action(t_idx, s_idx, step.action.clone());
                total_steps += 1;

                assert!(
                    app.session.failures.is_empty(),
                    "tutorial '{}' step '{}' produced parse failures: {:?}",
                    track.id,
                    step.title,
                    app.session.failures
                );
                assert!(
                    !app.session.evaluations.is_empty(),
                    "tutorial '{}' step '{}' left empty evaluations",
                    track.id,
                    step.title
                );
                assert!(
                    app.tutorial_state.completed_steps.contains(&(t_idx, s_idx)),
                    "tutorial '{}' step '{}' was not marked completed",
                    track.id,
                    step.title
                );

                // Scientific verification of key tutorial steps
                if track.id == "toxicity_alerts_lead_opt" && s_idx == 0 {
                    assert_eq!(app.session.evaluations.len(), 3);
                    let pains_hit = &app.session.evaluations[0];
                    assert!(
                        !pains_hit.medicinal_chemistry.pains_alerts.is_empty(),
                        "expected PAINS alerts on {}",
                        pains_hit.name
                    );
                    assert!(
                        !pains_hit.safety.dili_alerts.is_empty(),
                        "expected DILI alerts on {}",
                        pains_hit.name
                    );
                    let herg_hit = &app.session.evaluations[1];
                    assert_eq!(
                        herg_hit.safety.herg_risk,
                        HergRisk::High,
                        "expected High hERG risk on {}",
                        herg_hit.name
                    );
                    let clean_lead = &app.session.evaluations[2];
                    assert!(clean_lead.medicinal_chemistry.pains_alerts.is_empty());
                    assert!(clean_lead.medicinal_chemistry.brenk_alerts.is_empty());
                    assert!(clean_lead.safety.dili_alerts.is_empty());
                    assert_eq!(clean_lead.safety.herg_risk, HergRisk::Low);
                }

                if track.id == "batch_library_screening" && s_idx == 0 {
                    assert_eq!(app.session.evaluations.len(), 8);
                }
                if track.id == "batch_library_screening" && s_idx == 1 {
                    assert_eq!(
                        app.session.evaluations.len(),
                        4,
                        "expected 4 clean hits after filtering 8-compound library"
                    );
                    for ev in &app.session.evaluations {
                        assert_eq!(ev.druglikeness.lipinski.violations, 0);
                        assert!(ev.medicinal_chemistry.pains_alerts.is_empty());
                        assert!(ev.druglikeness.qed >= 0.50);
                        assert_ne!(ev.safety.herg_risk, HergRisk::High);
                    }
                }
            }
        }

        assert_eq!(app.tutorial_state.completed_steps.len(), total_steps);
    }
}
