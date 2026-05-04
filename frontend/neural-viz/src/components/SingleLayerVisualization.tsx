import { useAliases } from '../hooks/useAliases';
import SingleLayerView from './SingleLayerView';

export default function SingleLayerVisualization() {
  const { modelAlias, inputAlias, workAlias } = useAliases();
  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <SingleLayerView
        modelAlias={modelAlias} 
        inputAlias={inputAlias}
        workAlias={workAlias}
        layerId="layers.2" 
        pageDirection="TB" 
      />
    </div>
  );
}
