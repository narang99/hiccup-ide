import { useModelVisualization } from '../hooks/useModelVisualization';
import SharedCanvas from './SharedCanvas';

export default function ModelVisualization() {
  const {
    modelData,
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
  } = useModelVisualization();

  if (!modelData) {
    return <div className="flex items-center justify-center h-screen">Loading model...</div>;
  }

  return (
    <SharedCanvas
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      fitView
    />
  );
}