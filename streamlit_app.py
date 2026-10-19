import os
import io
import json
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

# Streamlit for Web UI
import streamlit as st

# PDF Extraction
import pypdf

# Google GenAI SDK (Official Gemini SDK)
from google import genai
from google.genai import types

class MaterialItem(BaseModel):
    name: str = Field(description="Bezeichnung des Materials oder Baugeräts")
    menge: str = Field(description="Menge inkl. Einheit (z.B. '25 to', '120 m²', '15 Säcke')")
    liefertermin_vorlauf: str = Field(description="Wann muss bestellt / angeliefert werden (z.B. 'Tag 1', '3 Tage vor Beginn')")
    anmerkung: Optional[str] = Field(None, description="Besonderheiten (z.B. Baustoffklasse, Entsorgungsnachweis, BigBags)")

class BehoerdenSchritt(BaseModel):
    behoerde_oder_stelle: str = Field(description="Zuständige Behörde oder Stelle (z.B. Gewerbeaufsichtsamt / BG BAU, Ordnungsamt)")
    anzeige_antrag: str = Field(description="Name des Antrags (z.B. TRGS 519 Anzeige, VAO / Halteverbot, SiGe-Plan)")
    frist_vorlauf: str = Field(description="Gesetzliche Frist (z.B. '7 Tage vor Beginn gem. TRGS 519', '14 Tage Vorlauf')")
    erforderliche_unterlagen: List[str] = Field(default_factory=list, description="Nachweise, Sachkundenachweise, Entsorgungskonzept")

class ArbeitsschutzSchadstoffe(BaseModel):
    psa_anforderungen: List[str] = Field(description="Erforderliche PSA (z.B. FFP3-Maske, Einwegoverall Kat. III Typ 5/6, Gebläsefilter)")
    schadstoff_art: Optional[str] = Field(None, description="Erkannte Schadstoffe (z.B. Asbest fest/schwach gebunden, KMF, PCB, PAK)")
    technische_massnahmen: List[str] = Field(description="Abschottung, Unterdruckhaltung, Schleusensystem, 4-Kammer-Schleuse, Befeuchtung")
    entsorgung_abfallschluessel: List[str] = Field(description="Vermutliche AVV-Schlüssel und Vorgaben (z.B. AVV 17 06 05* Asbest)")

class BaustellenLogistik(BaseModel):
    geraete_und_maschinen: List[str] = Field(description="Bagger, Meißel, Hebebühne, Container (Absetz-/Abrollmulden), Stromaggregat")
    baustelleneinrichtung: List[str] = Field(description="Bauzaun, Bauwasser, Baustrom, Sanitärcontainer / Schwarz-Weiß-Container")
    personal_einsatz: str = Field(description="Empfohlene Teameinteilung für die 7-Mann-Truppe inkl. Sachkundiger gem. TRGS 519 Anlage 3/4")

class BaustellenVorbereitungsPlan(BaseModel):
    projekt_titel: str = Field(description="Kurztitel des Projekts / Bauvorhabens")
    sparte: str = Field(description="Hochbau, Abbruch, Schadstoffsanierung oder Mischform")
    zusammenfassung: str = Field(description="Prägnanter Überblick über die anstehenden Arbeiten")
    behoerdliche_schritte: List[BehoerdenSchritt] = Field(description="Genehmigungen, Anzeigen und Fristen")
    schadstoff_arbeitsschutz: ArbeitsschutzSchadstoffe = Field(description="Spezifische Schutz- und Entsorgungsmaßnahmen")
    baustellen_logistik: BaustellenLogistik = Field(description="Einrichtung, Maschinen und 7-Mann Teamzuteilung")
    material_bestellliste: List[MaterialItem] = Field(description="Detaillierte Einkaufs- und Bestellliste")
    checkliste_vorlauf: List[str] = Field(description="Chronologische To-Do-Liste vor Baustelleneröffnung (T-14 bis T-0)")

def extract_text_from_pdf(uploaded_file) -> str:
    """Extrahiert den Rohtext aus einer hochgeladenen PDF-Datei."""
    pdf_reader = pypdf.PdfReader(uploaded_file)
    extracted_text = []
    for page_num, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            extracted_text.append(f"--- SEITE {page_num + 1} ---\n{text}")
    return "\n\n".join(extracted_text)

SYSTEM_PROMPT = """
Du bist ein erfahrener Bauleiter, Sachverständiger für Schadstoffsanierung (TRGS 519 / TRGS 521 / DGUV Regel 101-004) 
und Meister im Hochbau. Du arbeitest als leitender Assistent für ein hochspezialisiertes 7-Mann-Handwerksunternehmen.

Deine Aufgabe ist es, aus dem vorliegenden Leistungsverzeichnis (LV) alle notwendigen organisatorischen Schritte 
zur Baustellenvorbereitung sowie die komplette Material- und Gerätebestellung zu erstellen.

Berücksichtige strikt:
1. Teamgröße: Genau 7 Mitarbeiter stehen zur Verfügung. Teile die Ressourcen realistisch ein.
2. Vorschriften & Fristen:
   - Asbest/KMF: TRGS 519 / 521 (z.B. 7-Tage-Unternehmensanzeige an Bezirksregierung/Gewerbeaufsicht + BG Bau).
   - Schwarz-Weiß-Bereich, Dekontaminationsschleusen, Unterdruckhaltung, Entsorgungsnachweise (Begleitschein/Übernahmeschein).
   - Hochbau: Statische Sicherungen, Schalung, Bewehrungsabnahmen, Baustellensicherung (Verkehrsrechtliche Anordnung, Halteverbote).
3. Materialbestellung:
   - Exakte Auflistung benötigter Verbrauchsgüter (BigBags Asbest mit Warndruck, PE-Folien 0,2mm, Klebeband, Restfaserbindemittel, PSA).
   - Baustoffe für Hochbau (Zement, Kalksandstein, Dämmung etc.) inkl. Mengenangaben aus dem LV.
4. Sei extrem praxisnah, präzise und übersehe keine sicherheitskritischen Details.
"""

def analyze_lv_with_agent(pdf_text: str, api_key: str) -> BaustellenVorbereitungsPlan:
    """Übergibt das LV an Gemini mit Structured Outputs (Pydantic)."""
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
Analysiere das folgende Leistungsverzeichnis (LV) und erstelle den kompletten 
organisatorischen Baustellenvorbereitungsplan inklusive Materialbestellliste.

=== LEISTUNGSVERZEICHNIS TEXT START ===
{pdf_text}
=== LEISTUNGSVERZEICHNIS TEXT ENDE ===
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=BaustellenVorbereitungsPlan,
            temperature=0.2,
        )
    )
    
    parsed_result = BaustellenVorbereitungsPlan.model_validate_json(response.text)
    return parsed_result

def main():
    st.set_page_config(
        page_title="Bauleiter-Agent (Hochbau & Sanierung)",
        page_icon="🏗️",
        layout="wide"
    )

    st.title("🏗️ Digitaler Bauleiter: Baustellenvorbereitung & Materialplanung")
    st.markdown(
        "**Spezialisiert auf Hochbau, Abbruch & Schadstoffsanierung** | Ausgelegt für 7-Mann-Teams"
    )

    # Sidebar für Einstellungen & API-Key
    with st.sidebar:
        st.header("⚙️ Konfiguration")
        api_key = st.text_input(
            "Gemini API-Key eingeben",
            type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
            help="Hole dir einen Key unter https://aistudio.google.com/"
        )
        st.info("💡 **Sparten-Schwerpunkte:**\n- Asbest (TRGS 519)\n- KMF (TRGS 521)\n- Abbruch & Entkernung\n- Hochbau / Rohbau")

    uploaded_file = st.file_uploader("Leistungsverzeichnis (LV) als PDF hochladen", type=["pdf"])

    if uploaded_file is not None:
        st.success(f"Datei `{uploaded_file.name}` erfolgreich geladen.")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            start_btn = st.button("🚀 Baustelle vorbereiten", type="primary")

        if start_btn:
            if not api_key:
                st.error("Bitte gib zuerst einen gültigen Gemini API-Key in der Seitenleiste ein.")
                return

            with st.spinner("Lese PDF aus und analysiere Leistungsverzeichnis..."):
                try:
                    pdf_text = extract_text_from_pdf(uploaded_file)
                    if not pdf_text.strip():
                        st.error("Die PDF-Datei enthält keinen lesbaren Text (evtl. ein reiner Scan?). Bitte OCR durchführen.")
                        return

                    plan: BaustellenVorbereitungsPlan = analyze_lv_with_agent(pdf_text, api_key)
                    
                    st.toast("Analyse erfolgreich abgeschlossen!", icon="✅")
                    
                    st.divider()
                    st.header(f"📋 Vorbereitungsplan: {plan.projekt_titel}")
                    st.caption(f"Kategorie: {plan.sparte}")
                    st.markdown(f"**Projektüberblick:** {plan.zusammenfassung}")

                    tabs = st.tabs([
                        "📦 Material- & Bestellliste", 
                        "⚠️ Schadstoff & Arbeitsschutz", 
                        "🏛️ Behörden & Fristen", 
                        "🚜 Baustellenlogistik & 7-Mann Team", 
                        "📅 T-Minus Checkliste"
                    ])

                    # TAB 1: Materialbestellung
                    with tabs[0]:
                        st.subheader("Einkaufs- und Dispositionsliste")
                        st.markdown("Diese Materialien und Spezialausrüstungen müssen vorab disponiert werden:")
                        material_data = [
                            {
                                "Material / Gerät": m.name,
                                "Menge": m.menge,
                                "Liefertermin/Vorlauf": m.liefertermin_vorlauf,
                                "Hinweise / Normen": m.anmerkung or "-"
                            }
                            for m in plan.material_bestellliste
                        ]
                        st.dataframe(material_data, use_container_width=True)

                    # TAB 2: Schadstoffe & PSA
                    with tabs[1]:
                        st.subheader("Gefahrstoffmanagement & Arbeitssicherheit")
                        if plan.schadstoff_arbeitsschutz.schadstoff_art:
                            st.warning(f"**Identifizierte Schadstoffe:** {plan.schadstoff_arbeitsschutz.schadstoff_art}")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("#### 🦺 Erforderliche PSA")
                            for psa in plan.schadstoff_arbeitsschutz.psa_anforderungen:
                                st.markdown(f"- {psa}")

                            st.markdown("#### 🔬 Technische Schutzmaßnahmen")
                            for tech in plan.schadstoff_arbeitsschutz.technische_massnahmen:
                                st.markdown(f"- {tech}")

                        with col_b:
                            st.markdown("#### ♻️ Entsorgung & Abfallschlüssel (AVV)")
                            for avv in plan.schadstoff_arbeitsschutz.entsorgung_abfallschluessel:
                                st.markdown(f"- `{avv}`")

                    # TAB 3: Behörden
                    with tabs[2]:
                        st.subheader("Genehmigungen, Anzeigen & Vorlauffristen")
                        for beh in plan.behoerdliche_schritte:
                            with st.expander(f"📌 {beh.anzeige_antrag} ({beh.behoerde_oder_stelle})"):
                                st.markdown(f"**Frist / Vorlauf:** `{beh.frist_vorlauf}`")
                                st.markdown("**Benötigte Nachweise / Unterlagen:**")
                                for doc in beh.erforderliche_unterlagen:
                                    st.markdown(f"- {doc}")

                    # TAB 4: Logistik & 7-Mann-Team
                    with tabs[3]:
                        st.subheader("Baustelleninfrastruktur & Personaleinteilung")
                        st.info(f"👷 **Einsatzplan für 7-Mann-Truppe:**\n\n{plan.baustellen_logistik.personal_einsatz}")
                        
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.markdown("#### 🚜 Großgeräte & Maschinen")
                            for dev in plan.baustellen_logistik.geraete_und_maschinen:
                                st.markdown(f"- {dev}")
                        with col_m2:
                            st.markdown("#### 🚧 Baustelleneinrichtung (BE)")
                            for be in plan.baustellen_logistik.baustelleneinrichtung:
                                st.markdown(f"- {be}")

                    # TAB 5: Zeitplan / Checkliste
                    with tabs[4]:
                        st.subheader("Chronologische To-Do Liste (Vorlauf bis Baubeginn)")
                        for idx, item in enumerate(plan.checkliste_vorlauf, 1):
                            st.checkbox(f"{item}", key=f"check_{idx}")

                    st.divider()
                    json_str = plan.model_dump_json(indent=2)
                    st.download_button(
                        label="📥 Plan als JSON exportieren",
                        data=json_str,
                        file_name=f"Vorbereitung_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json"
                    )

                except Exception as e:
                    st.error(f"Fehler bei der Analyse: {str(e)}")

if __name__ == "__main__":
    main()
