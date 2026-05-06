import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Panel, type Node, type Edge, useNodesState, useEdgesState } from '@xyflow/react';
import { useAliases } from '../hooks/useAliases';
import { getHighActivatedPOIs, type HighActivatedPOIsResponse, type POIPoint, type UniqueActivationId } from '../fetchers/poi';
import SharedCanvas from './SharedCanvas';
import CosmeticButton from './shared/CosmeticButton';
import KernelLabelsManager from './shared/KernelLabelsManager';
import { makeEvenlySpacedLayout } from '../layouts';
import { type Direction } from '../types/direction';
import { type OverlayAlgorithm } from '../types/overlay';
import { DataTypeSelector } from './SharedCanvas/Controls/DataTypeSelector';
import { ColormapSelector } from './SharedCanvas/Controls/ColormapSelector';
import { DEFAULT_FETCHERS } from '../fetchers';

interface PoiPayloadForRender {
    activation: UniqueActivationId;
    point: POIPoint;
}

const getRectOverlayWithReceptiveField = (
    x: number, y: number, kernel_size: number, padding: number, stride: number
): OverlayAlgorithm => {
    const x_in_start = x * stride - padding;
    const y_in_start = y * stride - padding;
    const x_in_end = x_in_start + kernel_size;
    const y_in_end = y_in_start + kernel_size;

    return {
        type: 'DrawRect',
        start: [x_in_start, y_in_start],
        end: [x_in_end, y_in_end],
        color: "#f59e0b"
    }
}

const getActivationNode = (
    id: string,
    position: { x: number, y: number },
    title: string,
    coordinate: string,
    modelAlias: string,
    inputAlias: string,
    workAlias?: string,
    parentId?: string,
    width?: number,
    height?: number,
    overlayAlgorithm?: OverlayAlgorithm,
): Node => {
    return ({
        id: id,
        type: 'ActivationFlowNode',
        position: position,
        data: {
            coordinate: coordinate,
            fetchers: DEFAULT_FETCHERS,
            fetcherType: "activation",
            maxSize: 84,
            title: title,
            handleDirection: null,
            modelAlias,
            inputAlias,
            workAlias,
            overlayAlgorithm,
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

export default function HighlyActivatedPOIsPage() {
    const { nodeId, kernelIndex, inputIndex } = useParams<{ nodeId: string; kernelIndex: string; inputIndex: string }>();
    const navigate = useNavigate();
    const { modelAlias, inputAlias, workAlias } = useAliases();
    const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
    const [poiData, setPOIData] = useState<HighActivatedPOIsResponse | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const pageDirection: Direction = "TB";

    const weightCoordinate = nodeId && kernelIndex && inputIndex
        ? `${nodeId}.out_${kernelIndex}.in_${inputIndex}`
        : null;

    if (!weightCoordinate) {
        throw new Error("HighlyActivatedPOIsPage: Missing required route parameters (nodeId, kernelIndex, or inputIndex).");
    }

    // Hardcoded kernel parameters - match KernelSliceView
    const KERNEL_SIZE = 3;
    const STRIDE = 2;
    const PADDING = 1;

    // Fetch POI data
    useEffect(() => {
        if (modelAlias && weightCoordinate) {
            setIsLoading(true);
            getHighActivatedPOIs(modelAlias, weightCoordinate, 100) // Fetch more POIs for the dedicated page
                .then(setPOIData)
                .catch(console.error)
                .finally(() => setIsLoading(false));
        }
    }, [modelAlias, weightCoordinate]);

    // Generate nodes from POI data
    useEffect(() => {
        if (!poiData || poiData.pois.length === 0) {
            setNodes([]);
            setEdges([]);
            return;
        }

        const poiPayload: PoiPayloadForRender[] = [];
        poiData.pois.forEach(poi => {
            poi.input_activations.forEach(inputAct => {
                poiPayload.push({ activation: inputAct, point: poi.point });
            });
        });

        if (poiPayload.length === 0) {
            setNodes([]);
            setEdges([]);
            return;
        }

        const nodes: Node[] = [];
        const childWidth = 130;
        const childHeight = 150;
        const padding = 10;

        // Create a layout with more columns for better visualization
        const maxCols = Math.min(8, Math.ceil(Math.sqrt(poiPayload.length))); 
        const poiLayout = makeEvenlySpacedLayout(poiPayload.length, childHeight, childWidth, padding, pageDirection, maxCols);
        
        const poiLayerId = "poi-grid-layer";
        nodes.push({
            id: poiLayerId,
            type: 'LayerNode',
            position: { x: 0, y: 0 },
            width: poiLayout.parent.width,
            height: poiLayout.parent.height,
            data: {
                label: `Highly Activated POIs (${poiPayload.length} items)`,
                layerType: 'Input',
                nodeCount: poiPayload.length,
                handleDirection: null,
            },
        });

        // Add nodes for each POI
        poiPayload.forEach((payload, index) => {
            const inputAct = payload.activation;
            const point = payload.point;
            nodes.push(getActivationNode(
                `poi-${index}`,
                poiLayout.children[index],
                `${inputAct.input_alias} (${point.col}, ${point.row})`,
                inputAct.coordinate,
                inputAct.model_alias,
                inputAct.input_alias,
                undefined,
                poiLayerId,
                childWidth,
                childHeight,
                getRectOverlayWithReceptiveField(
                    point.col, point.row, KERNEL_SIZE, PADDING, STRIDE
                )
            ));
        });

        setNodes(nodes);
        setEdges([]);
    }, [poiData, setNodes, setEdges]);

    const handleBackClick = () => {
        navigate(`/models/${modelAlias}/${inputAlias}/${workAlias}/kernel-slice/${nodeId}/${kernelIndex}/${inputIndex}`);
    };

    if (isLoading) {
        return <div className="flex items-center justify-center h-screen">Loading highly activated POIs...</div>;
    }

    return (
        <div style={{ width: '100%', height: '100%', position: 'relative' }}>
            <SharedCanvas
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                fitView
                minZoom={0.1}
                maxZoom={2}
                pageDirection={pageDirection}
            >
                <Panel position="top-left" style={{ display: 'flex', gap: '8px' }}>
                    <CosmeticButton onClick={handleBackClick}>
                        ← Back to Kernel Slice
                    </CosmeticButton>
                    <KernelLabelsManager weightCoordinate={weightCoordinate} />
                </Panel>
                <Panel position="top-right" style={{ display: 'flex', flexDirection: 'column', gap: '10px', alignItems: 'flex-end' }}>
                    <DataTypeSelector />
                    <ColormapSelector />
                </Panel>
            </SharedCanvas>
        </div>
    );
}