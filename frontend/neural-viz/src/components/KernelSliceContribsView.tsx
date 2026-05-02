import { useParams, useNavigate } from 'react-router-dom';
import { Panel, type Node, type Edge } from '@xyflow/react';
import { type ModelData } from '../types/model';
import { DEFAULT_FETCHERS, type FetcherType } from '../fetchers';
import { useFetcherType } from '../hooks/useFetcherType';
import { useModelData } from '../hooks/useModelData';
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
): Node => {
    console.log("mera positionnnnnn", position, "id", id);
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

interface LayoutConfig {
    childWidth: number;
    childHeight: number;
    padding: number;
    direction: Direction;
}


const createSumGroup = (
    nodeId: string,
    kernelIdx: number,
    fetcherType: FetcherType,
    config: LayoutConfig,
): Node[] => {
    const sumLayerId = 'sum-layer';
    const sumLayout = makeEvenlySpacedLayout(1, config.childHeight, config.childWidth, config.padding, config.direction);

    return [
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
                handleDirection: config.direction,
            },
        },
        getNodeShowingActivation(
            'sum',
            sumLayout.children[0],
            "Sum",
            `${nodeId}.out_${kernelIdx}`,
            fetcherType,
            sumLayerId,
            config.childWidth,
            config.childHeight
        ),
    ];
};

const generateKernelSliceContribsView = (
    data: ModelData,
    nodeId: string,
    kernelIdx: number,
    fetcherType: FetcherType,
    pageDirection: Direction
): { nodes: Node[], edges: Edge[] } | null => {
    const targetNode = data.nodes.find(n => n.id === nodeId);

    if (!targetNode || targetNode.type !== 'Conv2d') return null;

    const inChannels = targetNode.params.in_channels as number;
    let nodes: Node[] = [];

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

    // const generateKernelSliceContribsView = useCallback((data: ModelData, nodeId: string, kernelIdx: number): { nodes: Node[], edges: Edge[] } | null => {
    //     const targetNode = data.nodes.find(n => n.id === nodeId);
    //     if (!targetNode || targetNode.type !== 'Conv2d') return null;

    //     const inChannels = targetNode.params.in_channels as number;
    //     let nodes: Node[] = [];

    //     const childWidth = 130;
    //     const childHeight = 150;
    //     const padding = 10;

    //     const sliceLayout = makeEvenlySpacedLayout(3, childHeight, childWidth, padding, pageDirection);
    //     const sliceParentLayerId =  "slice-parent";
    //     nodes.push(
    //         {
    //             id: sliceParentLayerId,
    //             type: 'LayerNode',
    //             position: { x: 0, y: 0 },
    //             width: sliceLayout.parent.width,
    //             height: sliceLayout.parent.height,
    //             data: {
    //                 label: "",
    //                 layerType: 'Conv2d',
    //                 nodeCount: inChannels,
    //                 handleDirection: pageDirection,
    //             },
    //         }
    //     )

    //     for (let i = 0; i < inChannels; i++) {
    //         nodes.push(getNodeShowingActivation(
    //             `output-${i}`,
    //             sliceLayout.children[i],
    //             "Output",
    //             `${nodeId}.out_${kernelIdx}.in_${i}`,
    //             fetcherType,
    //             sliceParentLayerId,
    //             childWidth,
    //             childHeight,
    //         ));
    //     }

    //     const sumConfig: LayoutConfig = {
    //         childWidth: 120,
    //         childHeight: 100,
    //         padding: 10,
    //         direction: pageDirection,
    //     };

    //     // 2. Create sum group
    //     nodes = nodes.concat(createSumGroup(nodeId, kernelIdx, fetcherType, sumConfig));


    //     const edges = [{
    //         id: "slice-to-sum",
    //         source: "slice-parent",
    //         target: "sum-layer",
    //         type: "default",
    //         style: { stroke: '#dc2626', strokeWidth: 2 },
    //     }]

    //     return { nodes, edges };
    // }, [fetcherType, pageDirection]);

    // const { kernelNodes, kernelEdges } = useMemo(() => {
    //     if (modelData && nodeId && kernelIndex) {
    //         const result = generateKernelSliceContribsView(modelData, nodeId, parseInt(kernelIndex), fetcherType, pageDirection);
    //         if (result) {
    //             return { kernelNodes: result.nodes, kernelEdges: result.edges };
    //         }
    //     }
    //     return { kernelNodes: [], kernelEdges: [] };
    // }, [modelData, nodeId, kernelIndex, fetcherType, pageDirection]);

    const handleBackClick = () => {
        navigate('/');
    };

    if (!modelData) {
        return <div className="flex items-center justify-center h-screen">Loading kernel details...</div>;
    }
    console.log("modellllllll", modelData);

    const isReady = modelData && nodeId && kernelIndex;
    const emptyNodes = { nodes: [], edges: [] }
    const result = isReady
        ? generateKernelSliceContribsView(modelData, nodeId, parseInt(kernelIndex), fetcherType, pageDirection)
        : emptyNodes;
    const { nodes, edges } = (result) ? result : emptyNodes;

    return (
        <SharedCanvas
            nodes={nodes}
            edges={edges}
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