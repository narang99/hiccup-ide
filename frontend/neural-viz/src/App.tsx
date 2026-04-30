import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ColormapProvider } from './contexts/ColormapContext';
import ModelVisualization from './components/ModelVisualization';
import KernelDetailView from './components/KernelDetailView';

function App() {
  return (
    <ColormapProvider>
      <Router>
        <Routes>
          <Route path="/" element={<ModelVisualization />} />
          <Route path="/kernel/:nodeId/:kernelIndex" element={<KernelDetailView />} />
        </Routes>
      </Router>
    </ColormapProvider>
  );
}

export default App;
