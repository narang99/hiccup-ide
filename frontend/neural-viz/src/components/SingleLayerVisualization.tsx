import SingleLayerView from './SingleLayerView';

export default function SingleLayerVisualization() {
  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <SingleLayerView
        modelAlias="example-model" 
        inputAlias="first-input"
        layerId="layers.2" 
        pageDirection="TB" 
      />
    </div>
  );
}
