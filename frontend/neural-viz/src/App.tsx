import { BrowserRouter as Router, Routes, Route, Outlet, Navigate } from 'react-router-dom';
import { ColormapProvider } from './contexts/ColormapContext';
import { FetcherTypeProvider } from './contexts/FetcherTypeContext';
import { AliasProvider } from './contexts/AliasContext';
import ModelVisualization from './components/ModelVisualization';
// import KernelDetailView from './components/KernelDetailView';
import KernelSliceContribsView from './components/KernelSliceContribsView';
import KernelSliceView from './components/KernelSliceView';
import HighlyActivatedPOIsPage from './components/HighlyActivatedPOIsPage';
import SingleLayerVisualization from './components/SingleLayerVisualization';
import PruneGraphView from './components/PruneGraphView';
import LandingPage from './components/LandingPage';

const AliasLayout = () => (
  <AliasProvider>
    <Outlet />
  </AliasProvider>
);

function App() {
  return (
    <FetcherTypeProvider>
      <ColormapProvider>
        <Router>
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/models/:modelAlias/:inputAlias/:workAlias" element={<AliasLayout />}>
              <Route index element={<ModelVisualization />} />
              <Route path="kernel/:nodeId/:kernelIndex" element={<KernelSliceContribsView />} />
              <Route path="kernel-slice/:nodeId/:kernelIndex/:inputIndex" element={<KernelSliceView />} />
              <Route path="poi-viewer/:nodeId/:kernelIndex/:inputIndex" element={<HighlyActivatedPOIsPage />} />
              <Route path="single-layer" element={<SingleLayerVisualization />} />
              <Route path="prune-graph/" element={<PruneGraphView />} />
            </Route>
            {/* Fallback for when aliases are missing - redirect to landing page */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Router>
      </ColormapProvider>
    </FetcherTypeProvider>
  );
}

export default App;
