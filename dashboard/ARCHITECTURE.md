"""
Vector Alpha Dashboard - Modular Architecture
==============================================

dashboard/
├── app.py                      # Main Streamlit entry point (minimal)
├── requirements.txt            # Dependencies (lean set)
├── config.py                   # Configuration constants
├── data_loader.py             # Centralized data loading with caching
├── components/
│   ├── __init__.py
│   ├── overview.py            # Project overview & KPIs
│   ├── performance.py         # Returns & performance metrics
│   ├── drawdown_risk.py       # Drawdown & rolling risk metrics
│   └── attribution.py         # Return & risk attribution
├── utils/
│   ├── __init__.py
│   ├── plotting.py            # Plotly visualization helpers
│   └── helpers.py             # General utility functions
└── pages/                      # (Future: multi-page layout)

KEY DESIGN PRINCIPLES
=============================================================================

1. DATA LOADING CENTRALIZED
   - data_loader.py handles all Parquet loading, validation, caching
   - No data manipulation in UI components
   - Fail loudly on missing files

2. COMPONENTS ARE PURE VISUALIZATION
   - No business logic
   - Accept DataFrames as parameters
   - Render Streamlit/Plotly objects only

3. CONFIGURATION ISOLATED
   - All constants in config.py
   - Asset names, date ranges, rolling windows

4. MODULAR PLOTTING
   - plotting.py contains all chart-creation functions
   - Reusable across components
   - Single responsibility (equity plot, returns plot, etc.)

5. APP ENTRY POINT MINIMAL
   - app.py is orchestrator only
   - ~50 lines: load data, dispatch to components
   - No UI rendering logic

DATA FLOW
=============================================================================

User Opens Dashboard
        v
app.py (orchestrator)
        v
data_loader.load_all_data()  [cached @st.cache_data]
        - prices.parquet
        - returns.parquet
        - drawdown.parquet
        - return_attribution.parquet
        - risk_attribution.parquet
        v
Sidebar: Select view + filters
        v
components/{overview|performance|drawdown_risk|attribution}.py
        - Accept DataFrames + filter params
        - No computation, only visualization
        - Call utils/plotting.py functions
        v
utils/plotting.py
        - Create Plotly figures (read-only data)
        - Return to component for st.plotly_chart()
        v
Display to User

RESPONSIBILITY MATRIX
=============================================================================

┌─────────────────────┬──────────────────────────────────────────────────────┐
│ File/Module         │ Responsibility                                       │
├─────────────────────┼──────────────────────────────────────────────────────┤
│ app.py              │ • Page config (title, layout)                        │
│                     │ • Load data via data_loader                          │
│                     │ • Sidebar navigation & filters                       │
│                     │ • Dispatch to components                             │
│                     │                                                      │
│ data_loader.py      │ • Load all Parquet files                             │
│                     │ • Cache with @st.cache_data                          │
│                     │ • Validate index alignment                           │
│                     │ • Fail loudly if missing                             │
│                     │                                                      │
│ config.py           │ • Constants (asset names, dates, windows)            │
│                     │ • File paths                                         │
│                     │ • Display settings                                   │
│                     │                                                      │
│ components/         │ • Pure visualization                                 │
│  overview.py        │ • Accept data, render Streamlit                      │
│  performance.py     │ • No business logic                                  │
│  drawdown_risk.py   │ • No data transformation                             │
│  attribution.py     │ • Call plotting.py for charts                        │
│                     │                                                      │
│ utils/              │ • Create Plotly figures                              │
│  plotting.py        │ • No Streamlit calls                                 │
│                     │ • Return fig objects                                 │
│                     │ • Single-responsibility functions                    │
│                     │                                                      │
│ utils/helpers.py    │ • Format dates, numbers                              │
│                     │ • Index validation                                   │
│                     │ • Reusable string/math utilities                     │
└─────────────────────┴──────────────────────────────────────────────────────┘

FILE SIZES (TARGET)
=============================================================================

app.py                  ~50 lines    (orchestrator only)
data_loader.py          ~80 lines    (loading + validation)
config.py               ~30 lines    (constants)
components/
  overview.py           ~40 lines    (KPIs + description)
  performance.py        ~50 lines    (returns histogram)
  drawdown_risk.py      ~80 lines    (3 plots + rolling selector)
  attribution.py        ~100 lines   (3 plots + filters)
utils/
  plotting.py           ~150 lines   (6-8 plot functions)
  helpers.py            ~40 lines    (utility functions)

TOTAL: ~620 lines (vs. 370+ in monolithic app.py, but MUCH cleaner)

ADVANTAGES OF MODULAR DESIGN
=============================================================================

+ Easy to test (components are pure functions)
+ Easy to extend (add new pages, new plots)
+ Easy to maintain (single responsibility)
+ Easy to debug (data flow is obvious)
+ Easy to reuse (plotting functions shared)
+ Easy to onboard (clear file purposes)
+ Separation of concerns (UI != logic != data)

"""
