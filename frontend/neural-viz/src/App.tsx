import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import ModelVisualization from './components/ModelVisualization';
import KernelDetailView from './components/KernelDetailView';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<ModelVisualization />} />
        <Route path="/kernel/:nodeId/:kernelIndex" element={<KernelDetailView />} />
      </Routes>
    </Router>
  );
}

export default App
