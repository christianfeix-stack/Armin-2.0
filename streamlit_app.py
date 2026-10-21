import os
import io
import json
import time
import tempfile
import traceback
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

# Streamlit für die Web UI
import streamlit as st

# Google GenAI SDK (Offizielles Gemini SDK)
from google import genai
from google.genai import types

# Google Drive API
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

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
    projektnummer: str = Field(description="Projektnummer aus dem LV. Falls nicht vorhanden, generiere Platzhalter z.B. 'P-UNBEKANNT'")
    auftraggeber: str = Field(description="Name des Auftraggebers aus dem LV. Falls nicht vorhanden, nutze 'Unbekannt'")
    projekt_titel: str = Field(description="Kurztitel des Projekts / Bauvorhabens")
    sparte: str = Field(description="Hochbau, Abbruch, Schadstoffsanierung oder Mischform")
    zusammenfassung: str = Field(description="Prägnanter Überblick über die anstehenden Arbeiten")
    behoerdliche_schritte: List[BehoerdenSchritt] = Field(description="Genehmigungen, Anzeigen und Fristen")
    schadstoff_arbeitsschutz: ArbeitsschutzSchadstoffe = Field(description="Spezifische Schutz- und Entsorgungsmaßnahmen")
    baustellen_logistik: BaustellenLogistik = Field(description="Einrichtung, Maschinen und 7-Mann Teamzuteilung")
    material_bestellliste: List[MaterialItem] = Field(description="Detaillierte Einkaufs- und Bestellliste")
    checkliste_vorlauf: List[str] = Field(description="Chronologische To-Do-Liste vor Baustelleneröffnung (T-14 bis T-0)")

SYSTEM_PROMPT = """
Du bist ein erfahrener Bauleiter, Sachverständiger für Schadstoffsanierung (TRGS 519 / TRGS 521 / DGUV Regel 101-004) 
und Meister im Hochbau. Du arbeitest als leitender Assistent für ein hochspezialisiertes 7-Mann-Handwerksunternehmen.

Deine Aufgabe ist es, aus dem vorliegenden Leistungsverzeichnis (LV) alle notwendigen organisatorischen Schritte 
zur Baustellenvorbereitung sowie die komplette Material- und Gerätebestellung zu erstellen. Lies gezielt Projektnummer und Auftraggeber aus.

Berücksichtige strikt:
1. Teamgröße: Genau 7 Mitarbeiter stehen zur Verfügung. Teile die Ressourcen realistisch ein.
2. Vorschriften & Fristen:
   - Asbest/KMF: TRGS 519 / 521.
   - Schwarz-Weiß-Bereich, Unterdruckhaltung, Entsorgungsnachweise.
   - Hochbau: Statische Sicherungen, Baustellensicherung.
3. Materialbestellung:
   - Exakte Auflistung benötigter Verbrauchsgüter inkl. Mengenangaben.
4. Sei extrem praxisnah, präzise und übersehe keine sicherheitskritischen Details.
"""

def extract_drive_id(folder_input: str) -> str:
    """Extrahiert die Drive-Ordner-ID, falls der Benutzer versehentlich den ganzen Link einfügt."""
    if "drive.google.com" in folder_input:
        return folder_input.split("/")[-1].split("?")[0]
    return folder_input.strip()

def analyze_lv_with_agent(uploaded_file, api_key: str) -> BaustellenVorbereitungsPlan:
    """Lädt die PDF zu Google Gemini hoch und wartet auf Verarbeitung."""
    client = genai.Client(api_key=api_key)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        gemini_file = client.files.upload(file=tmp_path)
        
        while gemini_file.state.name == "PROCESSING":
            time.sleep(2)
            gemini_file = client.files.get(name=gemini_file.name)
            
        if gemini_file.state.name == "FAILED":
            raise Exception("Google Server konnten diese Datei nicht verarbeiten.")

        prompt = "Analysiere das beigefügte Leistungsverzeichnis (LV) vollständig und erstelle den Baustellenvorbereitungsplan inklusive Projektnummer und Auftraggeber."

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[gemini_file, prompt],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=BaustellenVorbereitungsPlan,
                temperature=0.2,
            )
        )
        
        parsed_result = BaustellenVorbereitungsPlan.model_validate_json(response.text)
        return parsed_result

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def create_drive_folders(credentials_json_str: str, raw_folder_id: str, projektnummer: str, auftraggeber: str, projekt_titel: str):
    """Erstellt die gewünschte Ordnerstruktur in Google Drive."""
    try:
        parent_folder_id = extract_drive_id(raw_folder_id)
        
        creds_dict = json.loads(credentials_json_str)
        creds = Credentials.from_service_account_info(creds_dict, scopes=['https://www.googleapis.com/auth/drive'])
        service = build('drive', 'v3', credentials=creds)

        folder_name = f"{projektnummer}_{auftraggeber}_{projekt_titel}".replace("/", "-").replace("\\", "-")
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_folder_id]
        }
        main_folder = service.files().create(body=file_metadata, fields='id, webViewLink').execute()
        main_folder_id = main_folder.get('id')
        main_folder_link = main_folder.get('webViewLink')

        subfolders = [
            f"01_Organisation {projektnummer}",
            f"02_WS - LS {projektnummer}",
            f"03_BTB - Aufmaß {projektnummer}"
        ]
        
        for sub_name in subfolders:
            sub_metadata = {
                'name': sub_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [main_folder_id]
            }
            service.files().create(body=sub_metadata, fields='id').execute()

        return main_folder_link
    except Exception as e:
        return f"Fehler bei Google Drive: {str(e)}"

def main():
    st.set_page_config(
        page_title="Bauleiter Armin 2.0 (Hochbau & Sanierung)",
        page_icon="🏗️",
        layout="wide"
    )

    st.title("🏗️ Armin 2.0: Baustellenvorbereitung & Materialplanung")
    st.markdown("**Spezialisiert auf Hochbau, Abbruch & Schadstoffsanierung** | Ausgelegt für 7-Mann-Teams")

    with st.sidebar:
        st.header("⚙️ KI-Konfiguration")
        api_key = st.text_input(
            "Gemini API-Key eingeben",
            type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
            help="Hole dir einen Key unter https://aistudio.google.com/"
        )
        
        st.divider()
        st.header("☁️ Google Drive Automatisierung")
        drive_folder_id = st.text_input(
            "Ordner-ID oder Link von 'Baustellen'",
            help="Du kannst hier einfach den kompletten Link deines Ordners reinkopieren. Armin filtert die ID selbst heraus."
        )
        # ACHTUNG: Hier wurde das fehlerhafte type="password" entfernt!
        drive_creds = st.text_area(
            "Drive Service Account JSON",
            help="Füge hier den kompletten Text deiner Google Cloud credentials.json Datei ein."
        )

        st.info("💡 **Sparten-Schwerpunkte:**\n- Asbest (TRGS 519)\n- KMF (TRGS 521)\n- Abbruch & Hochbau\n- Ordner-Auto-Create")

    uploaded_file = st.file_uploader(
        "Leistungsverzeichnis (LV) als PDF hochladen (Limit 1GB)", 
        type=["pdf"]
    )

    if uploaded_file is not None:
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.last_uploaded_file = uploaded_file.name
            if "plan" in st.session_state:
                del st.session_state.plan

        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.success(f"Datei `{uploaded_file.name}` ({file_size_mb:.1f} MB) erfolgreich geladen.")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            start_btn = st.button("🚀 Baustelle vorbereiten", type="primary")

        if start_btn:
            if not api_key:
                st.error("Bitte gib zuerst einen gültigen Gemini API-Key in der Seitenleiste ein.")
            else:
                with st.spinner("Armin analysiert das Dokument... Das kann bei großen Scans bis zu 60 Sekunden dauern."):
                    try:
                        st.session_state.plan = analyze_lv_with_agent(uploaded_file, api_key)
                        st.toast("Analyse erfolgreich abgeschlossen!", icon="✅")
                    except Exception as e:
                        st.error(f"Fehler bei der Analyse: {str(e)}")

        if "plan" in st.session_state:
            plan = st.session_state.plan
            
            st.divider()
            st.header(f"📋 Projekt: {plan.projektnummer} - {plan.auftraggeber} - {plan.projekt_titel}")
            st.caption(f"Kategorie: {plan.sparte}")

            col_drive1, col_drive2 = st.columns([1, 3])
            with col_drive1:
                drive_btn = st.button("📂 Ablage in Google Drive anlegen", type="primary", use_container_width=True)
            
            if drive_btn:
                if not drive_folder_id or not drive_creds:
                    st.error("⚠️ Bitte trage links in der Seitenleiste zuerst den Ordner-Link und das Drive JSON ein.")
                else:
                    with st.spinner("Ordnerstruktur wird angelegt..."):
                        link = create_drive_folders(
                            drive_creds, 
                            drive_folder_id, 
                            plan.projektnummer, 
                            plan.auftraggeber, 
                            plan.projekt_titel
                        )
                        if link.startswith("http"):
                            st.success(f"✅ Ordnerstruktur fertig! [Hier klicken, um Ablage zu öffnen]({link})")
                        else:
                            st.error(link)
            
            st.markdown(f"**Projektüberblick:** {plan.zusammenfassung}")

            tabs = st.tabs([
                "📦 Material- & Bestellliste", 
                "⚠️ Schadstoff & Arbeitsschutz", 
                "🏛️ Behörden & Fristen", 
                "🚜 Baustellenlogistik & 7-Mann Team", 
                "📅 T-Minus Checkliste"
            ])

            with tabs[0]:
                st.subheader("Einkaufs- und Dispositionsliste")
                material_data = [{"Material / Gerät": m.name, "Menge": m.menge, "Vorlauf": m.liefertermin_vorlauf, "Hinweise": m.anmerkung or "-"} for m in plan.material_bestellliste]
                st.dataframe(material_data, use_container_width=True)

            with tabs[1]:
                st.subheader("Gefahrstoffmanagement & Arbeitssicherheit")
                if plan.schadstoff_arbeitsschutz.schadstoff_art:
                    st.warning(f"**Identifizierte Schadstoffe:** {plan.schadstoff_arbeitsschutz.schadstoff_art}")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("#### 🦺 Erforderliche PSA")
                    for psa in plan.schadstoff_arbeitsschutz.psa_anforderungen: st.markdown(f"- {psa}")
                    st.markdown("#### 🔬 Technische Maßnahmen")
                    for tech in plan.schadstoff_arbeitsschutz.technische_massnahmen: st.markdown(f"- {tech}")
                with col_b:
                    st.markdown("#### ♻️ Entsorgung (AVV)")
                    for avv in plan.schadstoff_arbeitsschutz.entsorgung_abfallschluessel: st.markdown(f"- `{avv}`")

            with tabs[2]:
                st.subheader("Genehmigungen, Anzeigen & Vorlauffristen")
                for beh in plan.behoerdliche_schritte:
                    with st.expander(f"📌 {beh.anzeige_antrag} ({beh.behoerde_oder_stelle})"):
                        st.markdown(f"**Frist / Vorlauf:** `{beh.frist_vorlauf}`")
                        for doc in beh.erforderliche_unterlagen: st.markdown(f"- {doc}")

            with tabs[3]:
                st.subheader("Baustelleninfrastruktur & Personaleinteilung")
                st.info(f"👷 **Einsatzplan für 7-Mann-Truppe:**\n\n{plan.baustellen_logistik.personal_einsatz}")
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.markdown("#### 🚜 Großgeräte & Maschinen")
                    for dev in plan.baustellen_logistik.geraete_und_maschinen: st.markdown(f"- {dev}")
                with col_m2:
                    st.markdown("#### 🚧 Baustelleneinrichtung (BE)")
                    for be in plan.baustellen_logistik.baustelleneinrichtung: st.markdown(f"- {be}")

            with tabs[4]:
                st.subheader("Chronologische To-Do Liste")
                for idx, item in enumerate(plan.checkliste_vorlauf, 1):
                    st.checkbox(f"{item}", key=f"check_{idx}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("🚨 Ein unerwarteter Fehler ist aufgetreten. Hier sind die technischen Details zur Fehlersuche:")
        st.code(traceback.format_exc(), language="python")
