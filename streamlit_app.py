import streamlit as st
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Import moduli del progetto
from inference.load_bundle import load_inference_bundle
from inference.data_source import DataSource, column_names
from inference.engine_manager import EngineManager
from inference.fleet_controller import FleetController

# --- COSTANTI E CONFIGURAZIONE ---
PAGE_TITLE = "Dashboard Manutenzione Predittiva"
DATA_PATH = os.path.join("model_training", "data", "test_FD001.txt")
UPDATE_DELAY = 1.5  # Secondi tra gli aggiornamenti automatici
SENSORS_TO_PLOT = ["sensor2", "sensor3", "sensor4", "sensor7", "sensor11", "sensor12"] # Selezione sensori significativi

st.set_page_config(
    page_title=PAGE_TITLE,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- FUNZIONI DI UTILITÀ ---

def load_data_and_initialize():
    """
    Carica il modello, lo scaler e i dati di test.
    Inizializza i componenti core del sistema.
    """
    try:
        # 1. Caricamento Bundle (Modello + Scaler)
        bundle = load_inference_bundle()
        
        # 2. Caricamento Dati Grezzi
        if not os.path.exists(DATA_PATH):
            st.error(f"File dati non trovato: {DATA_PATH}")
            st.stop()
            
        df_test = pd.read_csv(DATA_PATH, sep=" ", header=None)
        # Rimuove colonne vuote finali tipiche del dataset C-MAPSS
        df_test = df_test.iloc[:, :26]
        df_test.columns = column_names
        
        # 3. Inizializzazione Componenti
        data_source = DataSource(df_test, bundle.feature_cols)
        
        engine_manager = EngineManager(
            model=bundle.model,
            scaler=bundle.scaler,
            feature_cols=bundle.feature_cols,
            window_size=bundle.config["window_size"],
            device=bundle.device
        )
        
        controller = FleetController(data_source, engine_manager)
        
        return controller

    except Exception as e:
        st.error(f"Errore critico durante l'inizializzazione: {e}")
        st.stop()

def create_sensor_grid(df):
    """Genera una griglia di grafici Plotly per i sensori selezionati."""
    fig = make_subplots(
        rows=2, cols=3, 
        subplot_titles=SENSORS_TO_PLOT,
        vertical_spacing=0.15
    )
    
    for i, sensor_name in enumerate(SENSORS_TO_PLOT):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        if sensor_name in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['cycle'], 
                    y=df[sensor_name], 
                    mode='lines', 
                    name=sensor_name,
                    line=dict(width=1.5)
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text="Cicli", row=row, col=col, showgrid=True)
            fig.update_yaxes(showgrid=True, row=row, col=col)

    fig.update_layout(
        height=500, 
        showlegend=False,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def create_rul_chart(df):
    """Genera il grafico dell'andamento RUL."""
    fig = go.Figure()
    
    # Linea RUL predetta
    fig.add_trace(go.Scatter(
        x=df['cycle'], 
        y=df['rul_prediction'],
        mode='lines+markers',
        name='RUL Predetta',
        line=dict(color='firebrick', width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title="Andamento Vita Utile Residua (RUL)",
        xaxis_title="Cicli Operativi",
        yaxis_title="RUL (Cicli)",
        template="plotly_white",
        height=350,
        showlegend=True,
        hovermode="x unified"
    )
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    
    return fig

# --- LOGICA APPLICAZIONE ---

# 1. Gestione Stato Sessione
if "controller" not in st.session_state:
    with st.spinner("Avvio del sistema e caricamento modelli..."):
        st.session_state.controller = load_data_and_initialize()
        st.session_state.autorun = False

# 2. Intestazione
st.title(PAGE_TITLE)
st.markdown("Sistema di monitoraggio in tempo reale per flotta motori aeronautici.")

# 3. Barra dei Controlli
col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl_spacer = st.columns([1, 1, 1, 4])

with col_ctrl1:
    if st.button("Step Manuale", use_container_width=True):
        st.session_state.controller.step()

with col_ctrl2:
    label_autorun = "Ferma Simulazione" if st.session_state.autorun else "Avvia Simulazione"
    if st.button(label_autorun, use_container_width=True):
        st.session_state.autorun = not st.session_state.autorun

with col_ctrl3:
    if st.button("Reset Sistema", type="primary", use_container_width=True):
        st.session_state.controller.reset()
        st.session_state.autorun = False
        st.rerun()

st.divider()

# 4. Dashboard Principale
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Stato Flotta")
    fleet_data = st.session_state.controller.get_fleet_table()
    
    if fleet_data:
        df_fleet = pd.DataFrame(fleet_data)
        # rul_prediction può essere None finché il buffer non è pieno (window_size cicli).
        # Convertiamo a numerico: None/valori non parseabili -> NaN (evita crash nello Styler).
        if "rul_prediction" in df_fleet.columns:
            df_fleet["rul_prediction"] = pd.to_numeric(df_fleet["rul_prediction"], errors="coerce")
        # Formattazione condizionale per evidenziare RUL critiche
        st.dataframe(
            df_fleet.style.background_gradient(subset=['rul_prediction'], cmap='RdYlGn', vmin=0, vmax=150)
            .format({'rul_prediction': '{:.2f}'}, na_rep="—"),
            use_container_width=True,
            height=600
        )
    else:
        st.info("Nessun dato disponibile. Avvia la simulazione.")

with col_right:
    st.subheader("Dettaglio Motore")
    
    # Recupera gli ID disponibili dallo storico
    available_ids = sorted(list(st.session_state.controller.history.keys()))
    
    if available_ids:
        selected_id = st.selectbox("Seleziona ID Motore per analisi:", available_ids)
        
        # Recupera DataFrame storico specifico per il motore selezionato
        # Nota: richiede che FleetController abbia il metodo get_engine_history_df implementato
        engine_df = st.session_state.controller.get_engine_history_df(selected_id)
        
        if not engine_df.empty:
            # Grafico RUL
            st.plotly_chart(create_rul_chart(engine_df), use_container_width=True)
            
            # Grafico Sensori
            st.markdown("##### Telemetria Sensori")
            st.plotly_chart(create_sensor_grid(engine_df), use_container_width=True)
        else:
            st.warning("Dati insufficienti per il motore selezionato.")
    else:
        st.write("In attesa di dati dai motori...")

# 5. Logica Auto-Run
if st.session_state.autorun:
    time.sleep(UPDATE_DELAY)
    st.session_state.controller.step()
    st.rerun()