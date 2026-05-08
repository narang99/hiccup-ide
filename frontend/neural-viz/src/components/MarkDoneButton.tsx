import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getSliceStatus, updateSliceState, type SliceState } from "../fetchers";

interface MarkDoneButtonProps {
    modelAlias: string;
    inputAlias: string;
    workAlias: string;
    weightCoordinate: string;
}

const STATE_CYCLE: SliceState[] = ['not_done', 'skip', 'review', 'done'];

const getStateConfig = (state: SliceState) => {
    switch (state) {
        case 'not_done':
            return {
                label: 'Not Done',
                icon: '◯',
                background: 'rgba(13, 13, 20, 0.88)',
                color: 'white'
            };
        case 'skip':
            return {
                label: 'Skip',
                icon: '⚠',
                background: '#f59e0b',
                color: 'white'
            };
        case 'review':
            return {
                label: 'Review',
                icon: '👁',
                background: '#3b82f6',
                color: 'white'
            };
        case 'done':
            return {
                label: 'Done',
                icon: '✓',
                background: '#10b981',
                color: 'white'
            };
    }
};

export default function MarkDoneButton(
    { modelAlias, inputAlias, workAlias, weightCoordinate }: MarkDoneButtonProps
) {
    const queryClient = useQueryClient();

    // Query to get the current slice status
    const { data: sliceStatus, isLoading: isLoadingStatus } = useQuery({
        queryKey: ['slice-status', modelAlias, inputAlias, workAlias, weightCoordinate],
        queryFn: () => weightCoordinate ? getSliceStatus(modelAlias, inputAlias, workAlias, weightCoordinate) : null,
        enabled: !!weightCoordinate,
    });

    // Mutation to update slice state
    const updateStateMutation = useMutation({
        mutationFn: (newState: SliceState) => updateSliceState(modelAlias, inputAlias, workAlias, weightCoordinate!, newState),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['slice-status', modelAlias, inputAlias, workAlias, weightCoordinate] });
        },
    });

    const currentState = sliceStatus?.state ?? 'not_done';
    const stateConfig = getStateConfig(currentState);
    const isUpdating = updateStateMutation.isPending;

    const handleCycleState = () => {
        if (!weightCoordinate || isUpdating) return;

        const currentIndex = STATE_CYCLE.indexOf(currentState);
        const nextIndex = (currentIndex + 1) % STATE_CYCLE.length;
        const nextState = STATE_CYCLE[nextIndex];
        
        updateStateMutation.mutate(nextState);
    };

    return (
        <button
            onClick={handleCycleState}
            disabled={isUpdating || isLoadingStatus}
            style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '7px 12px',
                background: stateConfig.background,
                border: '1px solid rgba(255,255,255,0.09)',
                borderRadius: 10,
                backdropFilter: 'blur(10px)',
                boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
                color: stateConfig.color,
                fontSize: 11,
                fontWeight: 600,
                cursor: (isUpdating || isLoadingStatus) ? 'not-allowed' : 'pointer',
                opacity: (isUpdating || isLoadingStatus) ? 0.8 : 1,
                transition: 'background-color 0.2s ease',
            }}
        >
            {isUpdating
                ? 'Updating...'
                : `${stateConfig.icon} ${stateConfig.label}`
            }
        </button>
    );
}