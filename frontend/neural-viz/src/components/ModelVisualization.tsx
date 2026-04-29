import { ReactFlow, Background, Controls, MiniMap } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useModelVisualization } from '../hooks/useModelVisualization';

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
    <div style={{ width: '100vw', height: '100vh' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        fitView
      >
        <Controls />
        <MiniMap />
        <Background />
      </ReactFlow>
    </div>
  );
}