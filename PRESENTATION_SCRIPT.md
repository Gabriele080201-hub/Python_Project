# Script di Presentazione - Predictive Maintenance (6 minuti)

---

## SLIDE 1: PROJECT PRESENTATION

Buongiorno a tutti. Oggi vi presenterò il mio progetto per il corso di Programming in Python: un sistema di **Predictive Maintenance scalabile** che integra Deep Learning per la stima in tempo reale del Remaining Useful Life, ovvero la vita residua utile di motori aeronautici.

L'obiettivo è creare un'infrastruttura IoT in grado di processare dati sensoriali continui e fornire predizioni aggiornate in tempo reale, simulando quello che succederebbe in un ambiente industriale reale.

---

## SLIDE 2: Why a Notebook Is Not Enough

Prima di entrare nei dettagli tecnici, voglio spiegare **perché questo progetto va oltre un semplice notebook Jupyter**.

Normalmente, quando sviluppiamo modelli di machine learning, lavoriamo in notebook. Ed è perfetto per la fase di ricerca e sviluppo. Tuttavia, questi modelli hanno dei **limiti fondamentali** quando si tratta di applicazioni industriali reali.

Guardiamo **il problema** dal punto di vista tradizionale:
- I modelli vengono sviluppati in Jupyter notebooks
- L'esecuzione è offline ed episodica
- Le predizioni sono **statiche** - vengono generate una volta e non vengono continuamente aggiornate
- Di conseguenza, l'usabilità in contesti industriali è molto **bassa**

Come dicono nel settore: "Il modello da solo NON equivale a valore operazionale".

Ora, spostiamoci alla **prospettiva industriale**, quella che questo progetto vuole affrontare:
- I dati industriali arrivano **continuamente** da sensori IoT
- Le condizioni degli asset, come i motori, **evolvono nel tempo**
- Le decisioni di manutenzione dipendono dall'**ultima predizione disponibile**, non da una generata settimane fa
- Per questo, il modello deve essere **parte di un sistema in esecuzione continua**

Ecco perché l'obiettivo del progetto è progettare un **sistema Python modulare** per predizioni di RUL continue e stateful - un sistema che mantiene uno stato e si aggiorna costantemente.

---

## SLIDE 3: Key Problems to Solve

Per trasformare questa visione in realtà, ho dovuto affrontare **quattro vincoli chiave** che hanno guidato direttamente la progettazione dell'architettura.

**Primo vincolo**: Non abbiamo accesso a sensori reali. Quindi i dati sensoriali devono essere **simulati**, ma è fondamentale che il sistema sia progettato per funzionare allo stesso modo con input reali. L'architettura deve essere pronta per un eventuale deploy in produzione.

**Secondo vincolo**: Il modello richiede uno storico fisso di dati. Più precisamente, ogni predizione può essere effettuata solo dopo che sono disponibili **30 osservazioni consecutive**. Prima di raggiungere questa soglia, il sistema deve accumulare dati senza generare predizioni.

**Terzo vincolo**: Ogni motore evolve **indipendentemente**. Questo significa che ogni engine ha la propria storia e deve essere tracciato separatamente nel tempo. Non possiamo semplicemente mischiare i dati di motori diversi.

**Quarto vincolo**: Le predizioni devono essere **continuamente aggiornate**. Il modello non può essere eseguito una sola volta e poi dimenticato. Deve girare ripetutamente man mano che nuovi dati arrivano, aggiornando le stime di RUL in tempo reale.

Questi quattro vincoli hanno **guidato direttamente** la progettazione dell'architettura del sistema che vi mostrerò ora.

---

## SLIDE 4: System Architecture

L'architettura che sto sviluppando si basa su una **separazione chiara** tra quattro componenti: l'ingestione dei dati, la logica di controllo, la gestione dello stato e l'orchestrazione del sistema.

Vediamoli nel dettaglio, dal basso verso l'alto:

**Input Layer - DataSource**: Questo componente astrae i file CSV grezzi e simula stream IoT continui di sensori. La sua responsabilità è emettere eventi sensoriali step-by-step, come se i dati arrivassero da sensori fisici in tempo reale.

**Control Layer - EngineManager**: Questo è il cervello del sistema. Coordina la pipeline di processing, instrada gli eventi in arrivo verso il motore corretto, e **decide quando l'inferenza deve essere eseguita** - ricordate il vincolo delle 30 osservazioni? È qui che viene gestito.

**State Layer - EngineState**: Ogni motore ha il proprio stato. Questo componente mantiene lo stato per-engine, memorizzando i buffer delle sliding window - le finestre scorrevoli di 30 timestep - e tiene traccia delle predizioni storiche. È qui che risiede la "memoria" del sistema.

**Orchestration & UI - FleetController**: Infine, questo layer coordina DataSource e EngineManager, sincronizza la logica backend con il frontend Streamlit, e gestisce lo storico e la visualizzazione a livello di intera flotta di motori.

Questa architettura modulare garantisce che ogni componente abbia una responsabilità ben definita, rendendo il sistema **scalabile, testabile e manutenibile**.

---

## SLIDE 5: Requirements & Tech Stack

Per implementare questa architettura, sto utilizzando un **ecosistema Python robusto** progettato per analytics in tempo reale ad alte prestazioni e visualizzazione.

Le tecnologie chiave sono cinque:

**Streamlit** fornisce la dashboard interattiva e i controlli di simulazione. Permette di avviare, fermare, resettare la simulazione e vedere tutto in tempo reale.

**PyTorch** è il motore di inferenza di deep learning. Il modello che ho addestrato è una Spatial-Temporal Graph Neural Network Transformer, che combina Graph Neural Networks per imparare relazioni tra sensori, e Transformer per catturare pattern temporali.

**Pandas e NumPy** gestiscono i dati e il windowing temporale - quelle finestre scorrevoli di 30 timestep di cui parlavo prima.

**Plotly** crea visualizzazioni interattive della telemetria, permettendo di esplorare l'andamento di ogni sensore nel tempo.

E infine, **Joblib** permette il caricamento efficiente di scaler e metadata pre-addestrati, garantendo che i dati siano normalizzati correttamente prima dell'inferenza.

Tutto questo stack lavora insieme per fornire un sistema performante e user-friendly.

---

## SLIDE 6: Discussion & Suggestions

Questo conclude la panoramica del progetto. Ovviamente ci sono ancora **diverse aree da finalizzare** e migliorare.

Sono aperto a **suggerimenti** su:
- Come ottimizzare ulteriormente l'architettura
- Possibili estensioni del sistema, come l'aggiunta di alert automatici quando il RUL scende sotto una soglia critica
- Strategie di testing per garantire la robustezza del sistema
- Modalità di deployment in ambienti industriali reali

E naturalmente, sono pronto a rispondere a qualsiasi domanda o chiarimento sul progetto.

**Grazie per l'attenzione.**

---

**TIMING TOTALE: ~6 minuti (circa 850 parole)**
