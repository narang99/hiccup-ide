import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ColormapProvider } from './contexts/ColormapContext';
import { FetcherTypeProvider } from './contexts/FetcherTypeContext';
import ModelVisualization from './components/ModelVisualization';
// import KernelDetailView from './components/KernelDetailView';
import KernelSliceContribsView from './components/KernelSliceContribsView';
import SingleLayerVisualization from './components/SingleLayerVisualization';
import PruneGraphView from './components/PruneGraphView';

function App() {
  return (
    <FetcherTypeProvider>
      <ColormapProvider>
        <Router>
          <Routes>
            <Route path="/" element={<ModelVisualization />} />
            {/* <Route path="/kernel/:nodeId/:kernelIndex" element={<KernelDetailView />} /> */}
            <Route path="/kernel/:nodeId/:kernelIndex" element={<KernelSliceContribsView />} />
            <Route path="/single-layer" element={<SingleLayerVisualization />} />
            <Route path="/prune-graph/" element={<PruneGraphView />} />
          </Routes>
        </Router>
      </ColormapProvider>
    </FetcherTypeProvider>
  );
}

export default App;
