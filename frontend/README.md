# Volatility Arbitrage Dashboard - Frontend

Interactive React dashboard for Heston option pricing exploration and volatility arbitrage backtesting.

## Setup

### 1. Install Dependencies

```bash
cd frontend
npm install
```

This will install:
- React 18 + Vite (fast development)
- Plotly.js (interactive 2D/3D charts)
- Material-UI (components)
- Zustand (state management)
- Axios (API client)
- react-grid-layout (drag-and-drop panels)

### 2. Start Development Server

```bash
npm run dev
```

The dashboard will be available at [http://localhost:3000](http://localhost:3000)

**Note**: Make sure the backend is running on port 8000:
```bash
# In the project root:
cd /Users/willhammond/Adaptive\ Volatility\ Arbitrage\ Backtesting\ Engine
PYTHONPATH=. python3 backend/main.py
```

## Features

### Heston Explorer (Current)

- **Interactive Parameter Controls**: Adjust Heston model parameters with real-time updates
  - Initial variance (v₀)
  - Long-run variance (θ)
  - Mean reversion speed (κ)
  - Vol of vol (σᵥ)
  - Correlation (ρ)
  - Risk-free rate (r)
  - Spot price (S)

- **2D Heatmap**: Color-coded option prices across strikes and maturities
  - Pan and zoom
  - Hover for exact values
  - Export to PNG (1920x1080)

- **3D Surface**: Rotatable 3D visualization
  - Interactive camera controls
  - Contour projections
  - Export to PNG

- **Performance Metrics**:
  - Computation time display
  - Cache hit/miss indicator
  - Real-time updates with 500ms debouncing

### Coming Soon

- Backtest Dashboard
  - Equity curve with drawdown
  - Greeks evolution
  - Trade history
  - Performance metrics
- Drag-and-drop layout customization
- Dark mode

## Project Structure

```
frontend/
├── src/
│   ├── api/              # API client (Axios)
│   │   ├── client.js
│   │   └── hestonApi.js
│   ├── components/       # Reusable components
│   │   ├── Charts/
│   │   │   ├── HestonHeatmap.jsx
│   │   │   └── HestonSurface3D.jsx
│   │   └── Controls/
│   │       └── ParameterSlider.jsx
│   ├── pages/            # Page components
│   │   └── HestonExplorer/
│   │       └── index.jsx
│   ├── store/            # Zustand state management
│   │   └── hestonStore.js
│   ├── hooks/            # Custom React hooks
│   │   ├── useDebounce.js
│   │   └── useHestonPricing.js
│   ├── App.jsx           # Root component
│   └── main.jsx          # Entry point
├── package.json
└── vite.config.js
```

## Development

### Build for Production

```bash
npm run build
```

Output will be in `frontend/dist/`

### Preview Production Build

```bash
npm run preview
```

## API Integration

The frontend connects to the FastAPI backend at `http://localhost:8000`

**Endpoints used:**
- `POST /api/v1/heston/price-surface` - Compute option price surface
- `GET /api/v1/heston/cache/stats` - Get cache statistics
- `DELETE /api/v1/heston/cache` - Clear price surface cache

All API calls are debounced by 500ms to avoid excessive requests during parameter adjustments.

## Performance

- **Initial load**: ~500ms (with cache miss)
- **Parameter change**: <50ms (cache hit), ~150ms (cache miss)
- **Chart rendering**: 60fps for smooth interactions
- **Bundle size**: <500KB initial load (optimized with code splitting)

## Troubleshooting

**"Cannot connect to backend"**
- Ensure backend is running on port 8000
- Check CORS settings in `backend/main.py`

**"npm: command not found"**
- Install Node.js: https://nodejs.org/ or `brew install node`

**Charts not rendering**
- Check browser console for errors
- Ensure all dependencies installed: `npm install`
