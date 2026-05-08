import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
    getKernelLabels, 
    addKernelLabel, 
    removeKernelLabel,
    addLargerPattern,
    removeLargerPattern,
    type KernelLabelsResponse
} from '../fetchers/kernel_labels';
import PatternTagList from './shared/PatternTagList';
import PatternAddForm from './shared/PatternAddForm';

interface KernelLabelsDialogProps {
    isOpen: boolean;
    weightCoordinate: string;
    onClose: () => void;
}

export default function KernelLabelsDialog({ 
    isOpen, 
    weightCoordinate, 
    onClose 
}: KernelLabelsDialogProps) {
    const [newLabel, setNewLabel] = useState('');
    const [newLargerPattern, setNewLargerPattern] = useState('');
    const queryClient = useQueryClient();

    const { data: labelsData, isLoading } = useQuery({
        queryKey: ['kernelLabels', weightCoordinate],
        queryFn: () => getKernelLabels(weightCoordinate),
        enabled: isOpen && !!weightCoordinate,
        retry: false,
    });

    const labels = labelsData?.labels || ['spurious'];
    const largerPatterns = labelsData?.larger_patterns || [];

    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                onClose();
            }
        };

        if (isOpen) {
            document.addEventListener('keydown', handleKeyDown);
        }

        return () => {
            document.removeEventListener('keydown', handleKeyDown);
        };
    }, [isOpen, onClose]);

    const addLabelMutation = useMutation({
        mutationFn: (label: string) => addKernelLabel(weightCoordinate, label),
        onSuccess: (data) => {
            queryClient.setQueryData(['kernelLabels', weightCoordinate], (oldData: KernelLabelsResponse | undefined) => ({
                ...oldData,
                labels: data.labels
            } as KernelLabelsResponse));
            setNewLabel('');
        },
        onError: (error) => {
            console.error("Failed to add label:", error);
        }
    });

    const handleAddLabel = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!newLabel.trim() || labels.includes(newLabel.trim())) {
            return;
        }
        addLabelMutation.mutate(newLabel.trim());
    };

    const removeLabelMutation = useMutation({
        mutationFn: (label: string) => removeKernelLabel(weightCoordinate, label),
        onSuccess: (data) => {
            queryClient.setQueryData(['kernelLabels', weightCoordinate], (oldData: KernelLabelsResponse | undefined) => ({
                ...oldData,
                labels: data.labels
            } as KernelLabelsResponse));
        },
        onError: (error) => {
            console.error("Failed to remove label:", error);
        }
    });

    const addLargerPatternMutation = useMutation({
        mutationFn: (pattern: string) => addLargerPattern(weightCoordinate, pattern),
        onSuccess: (data) => {
            queryClient.setQueryData(['kernelLabels', weightCoordinate], (oldData: KernelLabelsResponse | undefined) => ({
                ...oldData,
                larger_patterns: data.larger_patterns
            } as KernelLabelsResponse));
            setNewLargerPattern('');
        },
        onError: (error) => {
            console.error("Failed to add larger pattern:", error);
        }
    });

    const removeLargerPatternMutation = useMutation({
        mutationFn: (pattern: string) => removeLargerPattern(weightCoordinate, pattern),
        onSuccess: (data) => {
            queryClient.setQueryData(['kernelLabels', weightCoordinate], (oldData: KernelLabelsResponse | undefined) => ({
                ...oldData,
                larger_patterns: data.larger_patterns
            } as KernelLabelsResponse));
        },
        onError: (error) => {
            console.error("Failed to remove larger pattern:", error);
        }
    });

    const handleRemoveLabel = async (labelToRemove: string) => {
        if (labelToRemove === 'spurious') {
            return;
        }
        removeLabelMutation.mutate(labelToRemove);
    };

    const handleAddLargerPattern = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!newLargerPattern.trim() || largerPatterns.includes(newLargerPattern.trim())) {
            return;
        }
        addLargerPatternMutation.mutate(newLargerPattern.trim());
    };

    const handleRemoveLargerPattern = async (patternToRemove: string) => {
        removeLargerPatternMutation.mutate(patternToRemove);
    };

    const handleClose = (e: React.MouseEvent) => {
        e.stopPropagation();
        onClose();
    };

    if (!isOpen) return null;

    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            backdropFilter: 'blur(4px)',
        }}>
            <div style={{
                background: '#1e1e2e',
                padding: '24px',
                borderRadius: '12px',
                border: '1px solid rgba(255,255,255,0.1)',
                width: '400px',
                boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5)',
            }}>
                <h3 style={{ margin: '0 0 16px 0', color: '#fff', fontSize: '16px' }}>
                    Manage Labels and Larger Patterns for Kernel
                </h3>
                
                {isLoading ? (
                    <div style={{ color: 'rgba(255,255,255,0.7)', padding: '20px', textAlign: 'center' }}>
                        Loading data...
                    </div>
                ) : (
                    <>
                        <PatternTagList
                            title="Current Patch Patterns"
                            patterns={labels}
                            onRemove={handleRemoveLabel}
                            isRemoving={removeLabelMutation.isPending}
                            color="#3b82f6"
                            preventRemoval={(pattern) => pattern === 'spurious'}
                        />

                        <PatternAddForm
                            label="Add New Patch Pattern"
                            value={newLabel}
                            onChange={setNewLabel}
                            onSubmit={handleAddLabel}
                            disabled={addLabelMutation.isPending}
                            placeholder="Enter new patch pattern..."
                            buttonColor="#10b981"
                            autoFocus={true}
                        />

                        <PatternTagList
                            title="Current Larger Patterns"
                            patterns={largerPatterns}
                            onRemove={handleRemoveLargerPattern}
                            isRemoving={removeLargerPatternMutation.isPending}
                            color="#f59e0b"
                            emptyMessage="No larger patterns defined"
                        />

                        <PatternAddForm
                            label="Add New Larger Pattern"
                            value={newLargerPattern}
                            onChange={setNewLargerPattern}
                            onSubmit={handleAddLargerPattern}
                            disabled={addLargerPatternMutation.isPending}
                            placeholder="Enter new larger pattern..."
                            buttonColor="#f59e0b"
                        />
                    </>
                )}

                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                    <button 
                        onClick={handleClose}
                        disabled={addLabelMutation.isPending || removeLabelMutation.isPending || addLargerPatternMutation.isPending || removeLargerPatternMutation.isPending}
                        style={{
                            padding: '8px 16px',
                            background: '#3b82f6',
                            border: 'none',
                            borderRadius: '6px',
                            color: '#fff',
                            cursor: 'pointer',
                            fontSize: '13px',
                            fontWeight: 600,
                            opacity: (addLabelMutation.isPending || removeLabelMutation.isPending || addLargerPatternMutation.isPending || removeLargerPatternMutation.isPending) ? 0.7 : 1,
                        }}
                    >
                        Done
                    </button>
                </div>
            </div>
        </div>
    );
}