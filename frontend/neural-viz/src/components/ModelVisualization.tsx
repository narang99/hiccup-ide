import { useModelVisualization } from '../hooks/useModelVisualization';
import SharedCanvas from './SharedCanvas';
import { useFetcherType } from '../hooks/useFetcherType';

export default function ModelVisualization() {
  const { fetcherType } = useFetcherType();
  const pageDirection = "TB";
  const {
    modelData,
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
  } = useModelVisualization(fetcherType, pageDirection);

  if (!modelData) {
    return <div className="flex items-center justify-center h-screen">Loading model...</div>;
  }

  return (
    <SharedCanvas
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      pageDirection={pageDirection}
      fitView
    />
  );
}