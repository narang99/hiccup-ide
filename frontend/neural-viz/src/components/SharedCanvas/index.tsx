import { type ReactNode } from 'react';
import { ReactFlow, Background, Controls, type ReactFlowProps } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useSelectedNodeStore } from '../../stores/selectedNodeStore';
import { LayerNode } from '../nodes/LayerNode';
import { ActivationFlowNode } from '../nodes/ActivationFlowNode';
import type { Direction } from '../../types/direction';

const nodeTypes = {
  LayerNode,
  ActivationFlowNode,
};

interface SharedCanvasProps extends ReactFlowProps {
  children?: ReactNode;
  pageDirection?: Direction;
}

export default function SharedCanvas({ children, pageDirection, ...props }: SharedCanvasProps) {
  const { handleNodeClick, handlePaneClick } = useSelectedNodeStore();

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <ReactFlow {...props} nodeTypes={nodeTypes} onNodeClick={handleNodeClick} onPaneClick={handlePaneClick} nodesDraggable={false}>
        <Controls />
        <Background />


        {children}
      </ReactFlow>
    </div>
  );
}
