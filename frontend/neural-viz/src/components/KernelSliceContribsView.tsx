import { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Panel, type Node, type Edge, useNodesState, useEdgesState } from '@xyflow/react';
import { type ModelData } from '../types/model';
import { DEFAULT_FETCHERS, type FetcherType } from '../fetchers';
import { useFetcherType } from '../hooks/useFetcherType';
import { useModelData } from '../hooks/useModelData';
import { useColormap } from '../hooks/useColormap';
import { useLayerStats } from '../hooks/useLayerStats';
import { updateNodeAbsMax } from '../utils/nodeUpdates';
import SharedCanvas from './SharedCanvas';
import { type HandleDirection } from './nodes/ActivationFlowNode';
import { makeEvenlySpacedLayout } from '../layouts';
import { toggleDirection, type Direction } from '../types/direction';

const getNodeShowingActivation = (
    id: string,
    position: { x: number, y: number },
    title: string,
    coordinate: string,
    fetcherType: FetcherType = "activation",
    parentId?: string,
    width?: number,
    height?: number,
    handleDirection: HandleDirection = null,
    absMax?: number,
): Node => {
    return ({
        id: id,
        type: 'ActivationFlowNode',
        position: position,
        data: {
            coordinate: coordinate,
            fetchers: DEFAULT_FETCHERS,
            fetcherType: fetcherType,
            maxSize: 84,
            title: title,
            handleDirection,
            absMax,
        },
        width: width,
        height: height,
        style: {
            background: 'transparent',
            border: '1px solid rgba(245, 158, 11, 0.35)',
            borderRadius: '6px',
            padding: 0,
            fontSize: '10px',
            overflow: 'hidden',
        },
        parentId: parentId,
        extent: parentId ? 'parent' : undefined,
    });
};




const generateKernelSliceContribsView = (
    data: ModelData,
    nodeId: string,
    kernelIdx: number,
    fetcherType: FetcherType,
    pageDirection: Direction,
    absMaxMap: Record<string, number> = {}
): { nodes: Node[], edges: Edge[] } | null => {
    const targetNode = data.nodes.find(n => n.id === nodeId);

    if (!targetNode || targetNode.type !== 'Conv2d') return null;

    const inChannels = targetNode.params.in_channels as number;
    const nodes: Node[] = [];

    const childWidth = 130;
    const childHeight = 150;
    const padding = 10;

    const directionInsideGroup = toggleDirection(pageDirection);
    const sliceLayout = makeEvenlySpacedLayout(inChannels, childHeight, childWidth, padding, directionInsideGroup);
    const sliceParentLayerId = "slice-parent";
    nodes.push(
        {
            id: sliceParentLayerId,
            type: 'LayerNode',
            position: { x: 0, y: 0 },
            width: sliceLayout.parent.width,
            height: sliceLayout.parent.height,
            data: {
                label: "",
                layerType: 'Conv2d',
                nodeCount: inChannels,
                handleDirection: pageDirection,
            },
        }
    )

    for (let i = 0; i < inChannels; i++) {
        nodes.push(getNodeShowingActivation(
            `output-${i}`,
            sliceLayout.children[i],
            "Output",
            `${nodeId}.out_${kernelIdx}.in_${i}`,
            fetcherType,
            sliceParentLayerId,
            childWidth,
            childHeight,
            null,
            absMaxMap[sliceParentLayerId]
        ));
    }

    const sumLayerId = 'sum-layer';
    const sumLayout = makeEvenlySpacedLayout(1, childHeight, childWidth, padding, directionInsideGroup);

    nodes.push(
        {
            id: sumLayerId,
            type: 'LayerNode',
            position: { x: 0, y: 0 },
            width: sumLayout.parent.width,
            height: sumLayout.parent.height,
            data: {
                label: 'Sum',
                layerType: 'Output',
                nodeCount: 1,
                handleDirection: pageDirection,
            }
        },
    )
    nodes.push(
        getNodeShowingActivation(
            'sum',
            sumLayout.children[0],
            "Sum",
            `${nodeId}.out_${kernelIdx}`,
            fetcherType,
            sumLayerId,
            childWidth,
            childHeight,
            null,
            absMaxMap[sumLayerId]
        )
    )


    const edges = [{
        id: "slice-to-sum",
        source: "slice-parent",
        target: "sum-layer",
        type: "default",
        style: { stroke: '#dc2626', strokeWidth: 2 },
    }]

    return { nodes, edges };
}

export default function KernelSliceContribsView() {
    const { fetcherType } = useFetcherType();
    const pageDirection: Direction = "TB";
    const { nodeId, kernelIndex } = useParams<{ nodeId: string; kernelIndex: string }>();
    const navigate = useNavigate();
    const { modelData } = useModelData("example-model");
    const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
    const [layerAbsMax, setLayerAbsMax] = useState<Record<string, number>>({});
    const { scalingMode } = useColormap();

    const generateNodesAndEdges = useCallback((
        data: ModelData,
        nodeId: string,
        kernelIdx: number,
        absMaxMap: Record<string, number> = {}
    ) => {
        const result = generateKernelSliceContribsView(data, nodeId, kernelIdx, fetcherType, pageDirection, absMaxMap);
        return result || { nodes: [], edges: [] };
    }, [fetcherType, pageDirection]);

    // Update nodes and edges when model data changes
    useEffect(() => {
        if (modelData && nodeId && kernelIndex) {
            const { nodes: newNodes, edges: newEdges } = generateNodesAndEdges(
                modelData,
                nodeId,
                parseInt(kernelIndex),
                {} // absMax will be applied separately to preserve selection state
            );
            setNodes(newNodes);
            setEdges(newEdges);
        }
    }, [modelData, nodeId, kernelIndex, generateNodesAndEdges, setNodes, setEdges, scalingMode]);

    // Separate effect for updating absMax to preserve selection state
    useEffect(() => {
        if (scalingMode === 'global' && Object.keys(layerAbsMax).length > 0) {
            setNodes((currentNodes) => updateNodeAbsMax(currentNodes, layerAbsMax));
        }
    }, [layerAbsMax, scalingMode, setNodes]);

    // Handle global scaling using custom hook
    useLayerStats({
        nodes,
        fetcherType,
        scalingMode,
        setLayerAbsMax,
    });

    const handleBackClick = () => {
        navigate('/');
    };

    if (!modelData) {
        return <div className="flex items-center justify-center h-screen">Loading kernel details...</div>;
    }

    return (
        <SharedCanvas
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            fitView
            minZoom={0.5}
            maxZoom={2}
            pageDirection={pageDirection}
        >
            {/* ── Back button ── */}
            <Panel position="top-left">
                <button
                    onClick={handleBackClick}
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 6,
                        padding: '7px 12px',
                        background: 'rgba(13, 13, 20, 0.88)',
                        border: '1px solid rgba(255,255,255,0.09)',
                        borderRadius: 10,
                        backdropFilter: 'blur(10px)',
                        boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
                        color: 'rgba(255,255,255,0.75)',
                        fontSize: 11,
                        fontWeight: 600,
                        cursor: 'pointer',
                    }}
                >
                    ← Overview
                </button>
            </Panel>
        </SharedCanvas>
    );
}