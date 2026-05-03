import SharedCanvas from './SharedCanvas';
import { useNodesState, useEdgesState } from '@xyflow/react';

export default function PruneGraphView() {
  const [nodes, onNodesChange] = useNodesState([]);
  const [edges, onEdgesChange] = useEdgesState([]);

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <SharedCanvas
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        pageDirection="TB"
      />
    </div>
  );
}
