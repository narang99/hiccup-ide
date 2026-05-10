import { useEffect, useState, useCallback, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Panel, type Node, type Edge, useNodesState, useEdgesState } from '@xyflow/react';
import { type ModelData } from '../types/model';
import { DEFAULT_FETCHERS, type FetcherType } from '../fetchers';
import { useModelData } from '../hooks/useModelData';
import SharedCanvas from './SharedCanvas';
import AnnotationDialog from './AnnotationDialog';
import CosmeticButton from './shared/CosmeticButton';
import CosmeticLink from './shared/CosmeticLink';
import KernelLabelsManager from './shared/KernelLabelsManager';
import { type HandleDirection } from './nodes/ActivationFlowNode';
import { makeEvenlySpacedLayout } from '../layouts';
import { type Direction } from '../types/direction';
import { DataTypeSelector } from './SharedCanvas/Controls/DataTypeSelector';
import { ColormapSelector } from './SharedCanvas/Controls/ColormapSelector';
import { type OverlayAlgorithm } from '../types/overlay';
import { getLayoutedLayerNodes } from '../layouts/layerLayout';

import { useAliases } from '../hooks/useAliases';
import { usePinnedWorkflows, generatePinnedWorkflowLinks } from '../hooks/usePinnedWorkflows';
import MarkDoneButton from './MarkDoneButton';
import KernelNotesWidget from './KernelNotesWidget';
import PinnedWorkflowsButton from './shared/PinnedWorkflowsButton';
import { getConvInputSliceStatus } from '../fetchers/sliceStatus';

const getActivationNode = (
    id: string,
    position: { x: number, y: number },
    title: string,
    coordinate: string,
    fetcherType: FetcherType = "activation",
    modelAlias: string,
    inputAlias: string,
    workAlias?: string,
    parentId?: string,
    width?: number,
    height?: number,
    handleDirection: HandleDirection = null,
    absMax?: number,
    link?: string,
    onPixelHover?: (nodeId: string, coordinate: string, gridCoord: [number, number], position: [number, number], value?: number) => void,
    onPixelLeave?: (nodeId: string, coordinate: string) => void,
    onPixelClick?: (nodeId: string, coordinate: string, gridCoord: [number, number] | null, position: [number, number] | null, value?: number) => void,
    overlayAlgorithm?: OverlayAlgorithm,
    showGreenIndicatorIfTrue?: () => Promise<boolean>,
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
            clickAction: link
                ? { type: 'link', link }
                : (onPixelClick ? { type: 'callback', callback: onPixelClick } : undefined),
            onPixelHover,
            onPixelLeave,
            modelAlias,
            inputAlias,
            workAlias,
            overlayAlgorithm,
            showGreenIndicatorIfTrue,
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

const getReceptiveFieldCoords = (
    x: number, y: number, kernel_size: number, padding: number, stride: number
): [[number, number], [number, number]] => {
    const x_in_start = x * stride - padding;
    const y_in_start = y * stride - padding;
    const x_in_end = x_in_start + kernel_size;
    const y_in_end = y_in_start + kernel_size;

    return [[x_in_start, y_in_start], [x_in_end, y_in_end]]
}

const getRectOverlayWithReceptiveField = (
    x: number, y: number, kernel_size: number, padding: number, stride: number
): OverlayAlgorithm => {
    const [start, end] = getReceptiveFieldCoords(x, y, kernel_size, padding, stride);
    return {
        type: 'DrawRect',
        rects: [{
            start: start,
            end: end,
        }],
    }
}


const generateKernelSliceView = (
    data: ModelData,
    nodeId: string,
    kernelIdx: number,
    inputIdx: number,
    pageDirection: Direction,
    modelAlias: string,
    inputAlias: string,
    workAlias: string,
    onPixelHover?: (nodeId: string, coordinate: string, gridCoord: [number, number], position: [number, number], value?: number) => void,
    onPixelLeave?: (nodeId: string, coordinate: string) => void,
    onPixelClick?: (nodeId: string, coordinate: string, gridCoord: [number, number] | null, position: [number, number] | null, value?: number) => void,
    inputOverlay?: OverlayAlgorithm,
    onFinalActPixelClick?: (nodeId: string, coordinate: string, gridCoord: [number, number] | null, position: [number, number] | null, value?: number) => void,
): { nodes: Node[], edges: Edge[] } | null => {
    const targetNode = data.nodes.find(n => n.id === nodeId);
    if (!targetNode || targetNode.type !== 'Conv2d') return null;

    const targetEdge = data.edges.find(e => e.target === nodeId);
    const inputNodeId = targetEdge ? targetEdge.source : "x";

    const nodes: Node[] = [];
    const childWidth = 130;
    const childHeight = 150;
    const padding = 10;

    // 1. Input Activation Layer
    const inputLayout = makeEvenlySpacedLayout(1, childHeight, childWidth, padding, "TB");
    const inputLayerId = "input-layer";
    nodes.push({
        id: inputLayerId,
        type: 'LayerNode',
        position: { x: 0, y: 0 },
        width: inputLayout.parent.width,
        height: inputLayout.parent.height,
        data: {
            label: "Input Activation",
            layerType: 'Input',
            nodeCount: 1,
            handleDirection: pageDirection,
        },
    });
    const inputActNode = getActivationNode(
        "input-act",
        inputLayout.children[0],
        `Channel ${inputIdx}`,
        `${inputNodeId}.out_${inputIdx}`,
        "activation",
        modelAlias,
        inputAlias,
        workAlias,
        inputLayerId,
        childWidth,
        childHeight,
        null
    );
    inputActNode.data = { ...inputActNode.data, overlayAlgorithm: inputOverlay };
    nodes.push(inputActNode);

    // 2. Weight Layer
    const weightLayout = makeEvenlySpacedLayout(1, childHeight, childWidth, padding, "TB");
    const weightLayerId = "weight-layer";
    nodes.push({
        id: weightLayerId,
        type: 'LayerNode',
        position: { x: 0, y: 0 },
        width: weightLayout.parent.width,
        height: weightLayout.parent.height,
        data: {
            label: "Kernel Weight",
            layerType: 'Conv2d',
            nodeCount: 1,
            handleDirection: pageDirection,
        },
    });
    nodes.push(getActivationNode(
        "weight-node",
        weightLayout.children[0],
        `W[${kernelIdx}][${inputIdx}]`,
        `${nodeId}.out_${kernelIdx}.in_${inputIdx}`,
        "weight",
        modelAlias,
        inputAlias,
        workAlias,
        weightLayerId,
        childWidth,
        childHeight,
        null
    ));

    // 3. Output Contribution Layer (Activation and Saliency)
    const outputLayout = makeEvenlySpacedLayout(2, childHeight, childWidth, padding, "TB");
    const outputLayerId = "output-layer";
    nodes.push({
        id: outputLayerId,
        type: 'LayerNode',
        position: { x: 0, y: 0 },
        width: outputLayout.parent.width,
        height: outputLayout.parent.height,
        data: {
            label: "Output Contribution",
            layerType: 'Conv2d',
            nodeCount: 2,
            handleDirection: pageDirection,
        },
    });
    nodes.push(getActivationNode(
        "output-act",
        outputLayout.children[0],
        "Activation",
        `${nodeId}.out_${kernelIdx}.in_${inputIdx}`,
        "activation",
        modelAlias,
        inputAlias,
        workAlias,
        outputLayerId,
        childWidth,
        childHeight,
        null,
        undefined,
        undefined,
        onPixelHover,
        onPixelLeave
    ));
    nodes.push(getActivationNode(
        "output-saliency",
        outputLayout.children[1],
        "Saliency",
        `${nodeId}.out_${kernelIdx}.in_${inputIdx}`,
        "saliency_map",
        modelAlias,
        inputAlias,
        workAlias,
        outputLayerId,
        childWidth,
        childHeight,
        null,
        undefined,
        undefined,
        onPixelHover,
        onPixelLeave,
        onPixelClick,
        undefined,
        async () => {
            const coordinate = `${nodeId}.out_${kernelIdx}.in_${inputIdx}`;
            const status = await getConvInputSliceStatus(modelAlias, inputAlias, workAlias, coordinate);
            return status.state === 'done';
        }
    ));

    // 4. Final Activation Layer
    const finalActLayout = makeEvenlySpacedLayout(1, childHeight, childWidth, padding, "TB");
    const finalActLayerId = "final-activation-layer";
    nodes.push({
        id: finalActLayerId,
        type: 'LayerNode',
        position: { x: 0, y: 0 },
        width: finalActLayout.parent.width,
        height: finalActLayout.parent.height,
        data: {
            label: "Kernel Activation",
            layerType: 'Conv2d',
            nodeCount: 1,
            handleDirection: pageDirection,
        },
    });
    nodes.push(getActivationNode(
        "final-act",
        finalActLayout.children[0],
        "Final Activation",
        `${nodeId}.out_${kernelIdx}`,
        "activation",
        modelAlias,
        inputAlias,
        workAlias,
        finalActLayerId,
        childWidth,
        childHeight,
        null,
        undefined,
        undefined,
        onPixelHover,
        onPixelLeave,
        onFinalActPixelClick
    ));


    const edges: Edge[] = [
        {
            id: "input-to-weight",
            source: inputLayerId,
            target: weightLayerId,
            type: "default",
            style: { stroke: '#94a3b8', strokeWidth: 2 },
        },
        {
            id: "weight-to-output",
            source: weightLayerId,
            target: outputLayerId,
            type: "default",
            style: { stroke: '#dc2626', strokeWidth: 2 },
        },
        {
            id: "output-to-final",
            source: outputLayerId,
            target: finalActLayerId,
            type: "default",
            style: { stroke: '#94a3b8', strokeWidth: 2 },
        }
    ];


    return { nodes, edges };
};

export default function KernelSliceView() {
    const { nodeId, kernelIndex, inputIndex } = useParams<{ nodeId: string; kernelIndex: string; inputIndex: string }>();
    const navigate = useNavigate();
    const { modelAlias, inputAlias, workAlias } = useAliases();
    const { modelData } = useModelData(modelAlias);
    const { data: pinnedWorks = [], isLoading: pinnedWorksLoading } = usePinnedWorkflows(modelAlias);
    const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
    const [inputOverlay, setInputOverlay] = useState<OverlayAlgorithm>({ type: 'NoOverlay' });
    const [hoveredValue, setHoveredValue] = useState<number | null>(null);
    const pageDirection: Direction = "LR";

    const weightCoordinate = nodeId && kernelIndex && inputIndex
        ? `${nodeId}.out_${kernelIndex}.in_${inputIndex}`
        : null;

    if (!weightCoordinate) {
        throw new Error("KernelSliceView: Missing required route parameters (nodeId, kernelIndex, or inputIndex) to determine weightCoordinate.");
    }

    // Annotation Dialog state
    const [isDialogOpen, setIsDialogOpen] = useState(false);
    const [dialogInfo, setDialogInfo] = useState<{ gridCoord: [number, number] | null } | null>(null);
    
    // Receptive field params hardcoded: ((3,3), 1, 2, 0)
    // kernel_size, stride, padding, dilation
    const KERNEL_SIZE = 3;
    const STRIDE = 2;
    const PADDING = 1;

    const handlePixelHover = useCallback((id: string, _: string, gridCoord: [number, number], _position: [number, number], value?: number) => {
        const [x, y] = gridCoord;
        if (id === 'output-act' || id === 'output-saliency' || id === 'final-act') {
            setInputOverlay(
                getRectOverlayWithReceptiveField(x, y, KERNEL_SIZE, PADDING, STRIDE)
            );
            setHoveredValue(value ?? null);
        }
    }, [STRIDE, PADDING, KERNEL_SIZE]);

    const handlePixelLeave = useCallback(() => {
        setInputOverlay({ type: 'NoOverlay' });
        setHoveredValue(null);
    }, []);

    const handlePixelClick = useCallback((_: string, __: string, gridCoord: [number, number] | null) => {
        setDialogInfo({ gridCoord });
        setIsDialogOpen(true);
    }, []);

    const handleFinalActPixelClick = useCallback((_id: string, _coord: string, gridCoord: [number, number] | null) => {
        if (!gridCoord) return;
        const [x, y] = gridCoord;
        const params = new URLSearchParams({
            type: 'Conv2dOutputCoordinate',
            layer_name: nodeId!,
            channel: kernelIndex!,
            y: y.toString(),
            x: x.toString(),
            layer_type: 'conv2d',
            coordinate_type: 'output'
        });
        window.open(`/models/${modelAlias}/${inputAlias}/${workAlias}/ui-graph/?${params.toString()}`, '_blank');
    }, [modelAlias, inputAlias, workAlias, nodeId, kernelIndex]);


    useEffect(() => {
        if (modelData && nodeId && kernelIndex && inputIndex) {
            const result = generateKernelSliceView(
                modelData,
                nodeId,
                parseInt(kernelIndex),
                parseInt(inputIndex),
                pageDirection,
                modelAlias,
                inputAlias,
                workAlias,
                handlePixelHover,
                handlePixelLeave,
                handlePixelClick,
                inputOverlay,
                handleFinalActPixelClick
            );
            if (result) {
                setNodes(result.nodes);
                setEdges(result.edges);
            }
        }
    }, [modelData, nodeId, kernelIndex, inputIndex, setNodes, setEdges, handlePixelHover, handlePixelLeave, handlePixelClick, inputOverlay, modelAlias, inputAlias, workAlias, handleFinalActPixelClick]);

    const handleBackClick = () => {
        navigate(`/models/${modelAlias}/${inputAlias}/${workAlias}/kernel/${nodeId}/${kernelIndex}`);
    };

    const pinnedWorkflowLinks = nodeId && kernelIndex && inputIndex 
        ? generatePinnedWorkflowLinks(pinnedWorks, modelAlias, nodeId, kernelIndex, inputIndex, inputAlias, workAlias)
        : [];


    const { nodes: layoutedNodes, edges: layoutedEdges } = useMemo(() => 
        getLayoutedLayerNodes(nodes, edges, pageDirection),
        [nodes, edges, pageDirection]
    );

    if (!modelData) {
        return <div className="flex items-center justify-center h-screen">Loading slice details...</div>;
    }

    return (
        <div style={{ width: '100%', height: '100%', position: 'relative' }}>
            <SharedCanvas
                nodes={layoutedNodes}
                edges={layoutedEdges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                fitView
                minZoom={0.5}
                maxZoom={2}
                pageDirection={pageDirection}
            >
                <Panel position="top-left" style={{ display: 'flex', gap: '8px' }}>
                    <CosmeticButton onClick={handleBackClick}>
                        ← Kernel View
                    </CosmeticButton>
                    <KernelLabelsManager weightCoordinate={weightCoordinate} />
                    <MarkDoneButton
                        modelAlias={modelAlias}
                        inputAlias={inputAlias}
                        workAlias={workAlias}
                        weightCoordinate={weightCoordinate}
                    />
                    <PinnedWorkflowsButton 
                        links={pinnedWorkflowLinks}
                        loading={pinnedWorksLoading}
                    />
                    <CosmeticLink 
                        to={`/models/${modelAlias}/${inputAlias}/${workAlias}/poi-viewer/${nodeId}/${kernelIndex}/${inputIndex}`}
                        target="_blank"
                    >
                        👁️ View All POIs
                    </CosmeticLink>
                    <div style={{
                        padding: '4px 8px',
                        background: 'rgba(0, 0, 0, 0.1)',
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        borderRadius: '4px',
                        fontSize: '12px',
                        color: '#fff',
                        minWidth: '60px',
                        textAlign: 'center'
                    }}>
                        {hoveredValue !== null ? `${hoveredValue.toFixed(4)}` : '—'}
                    </div>
                </Panel>
                <Panel position="top-right" style={{ display: 'flex', flexDirection: 'column', gap: '10px', alignItems: 'flex-end' }}>
                    <DataTypeSelector />
                    <ColormapSelector />
                </Panel>
            </SharedCanvas>

            <AnnotationDialog
                isOpen={isDialogOpen}
                gridCoord={dialogInfo?.gridCoord ?? null}
                workAlias={workAlias}
                weightCoordinate={weightCoordinate as string}
                onClose={() => setIsDialogOpen(false)}
            />

            <KernelNotesWidget weightCoordinate={weightCoordinate as string} />

        </div>
    );
}
