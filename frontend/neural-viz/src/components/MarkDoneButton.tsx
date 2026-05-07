import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getSliceStatus, markSliceDone, unmarkSliceDone } from "../fetchers";

interface MarkDoneButtonProps {
    modelAlias: string;
    inputAlias: string;
    workAlias: string;
    weightCoordinate: string;
}

export default function MarkDoneButton(
    { modelAlias, inputAlias, workAlias, weightCoordinate }: MarkDoneButtonProps
) {
    const queryClient = useQueryClient();

    // Query to get the current done status
    const { data: sliceStatus, isLoading: isLoadingStatus } = useQuery({
        queryKey: ['slice-status', modelAlias, inputAlias, workAlias, weightCoordinate],
        queryFn: () => weightCoordinate ? getSliceStatus(modelAlias, inputAlias, workAlias, weightCoordinate) : null,
        enabled: !!weightCoordinate,
    });

    // Mutation to mark slice as done
    const markDoneMutation = useMutation({
        mutationFn: () => markSliceDone(modelAlias, inputAlias, workAlias, weightCoordinate!),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['slice-status', modelAlias, inputAlias, workAlias, weightCoordinate] });
        },
    });

    // Mutation to unmark slice as done
    const unmarkDoneMutation = useMutation({
        mutationFn: () => unmarkSliceDone(modelAlias, inputAlias, workAlias, weightCoordinate!),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['slice-status', modelAlias, inputAlias, workAlias, weightCoordinate] });
        },
    });

    const isDone = sliceStatus?.is_done ?? false;
    const isMarking = markDoneMutation.isPending || unmarkDoneMutation.isPending;

    const handleToggleDone = () => {
        if (!weightCoordinate || isMarking) return;

        if (isDone) {
            unmarkDoneMutation.mutate();
        } else {
            markDoneMutation.mutate();
        }
    };

    return (
        <button
            onClick={handleToggleDone}
            disabled={isMarking || isLoadingStatus}
            style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '7px 12px',
                background: isDone ? '#10b981' : 'rgba(13, 13, 20, 0.88)',
                border: '1px solid rgba(255,255,255,0.09)',
                borderRadius: 10,
                backdropFilter: 'blur(10px)',
                boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
                color: 'white',
                fontSize: 11,
                fontWeight: 600,
                cursor: (isMarking || isLoadingStatus) ? 'not-allowed' : 'pointer',
                opacity: (isMarking || isLoadingStatus) ? 0.8 : 1,
            }}
        >
            {isMarking
                ? (isDone ? 'Unmarking...' : 'Marking...')
                : isDone
                    ? '✓ Done'
                    : '✓ Not Done'
            }
        </button>
    );
}