import { useState, useRef, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
    getKernelLabels, 
    addKernelLabel, 
    removeKernelLabel,
    type KernelLabelsResponse
} from '../fetchers/kernel_labels';

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
    const queryClient = useQueryClient();
    const inputRef = useRef<HTMLInputElement>(null);

    const { data: labelsData, isLoading } = useQuery({
        queryKey: ['kernelLabels', weightCoordinate],
        queryFn: () => getKernelLabels(weightCoordinate),
        enabled: isOpen && !!weightCoordinate,
        retry: false,
    });

    const labels = labelsData?.labels || ['spurious'];

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
            inputRef.current?.focus();
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

    const handleRemoveLabel = async (labelToRemove: string) => {
        if (labelToRemove === 'spurious') {
            return;
        }
        removeLabelMutation.mutate(labelToRemove);
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
                    Manage Labels for Kernel
                </h3>
                
                {isLoading ? (
                    <div style={{ color: 'rgba(255,255,255,0.7)', padding: '20px', textAlign: 'center' }}>
                        Loading labels...
                    </div>
                ) : (
                    <>
                        <div style={{ marginBottom: '20px' }}>
                            <h4 style={{ margin: '0 0 12px 0', color: 'rgba(255,255,255,0.8)', fontSize: '14px' }}>
                                Current Labels:
                            </h4>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                                {labels.map(label => (
                                    <div
                                        key={label}
                                        style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '6px',
                                            background: label === 'spurious' ? '#374151' : '#3b82f6',
                                            color: '#fff',
                                            padding: '4px 8px',
                                            borderRadius: '16px',
                                            fontSize: '12px',
                                            fontWeight: 500,
                                        }}
                                    >
                                        {label}
                                        {label !== 'spurious' && (
                                            <button
                                                onClick={() => handleRemoveLabel(label)}
                                                disabled={removeLabelMutation.isPending}
                                                style={{
                                                    background: 'transparent',
                                                    border: 'none',
                                                    color: 'rgba(255,255,255,0.8)',
                                                    cursor: 'pointer',
                                                    fontSize: '14px',
                                                    padding: 0,
                                                    marginLeft: '2px',
                                                }}
                                            >
                                                ×
                                            </button>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>

                        <form onSubmit={handleAddLabel} style={{ marginBottom: '24px' }}>
                            <label style={{ 
                                display: 'block', 
                                color: 'rgba(255,255,255,0.6)', 
                                fontSize: '12px', 
                                marginBottom: '4px' 
                            }}>
                                Add New Label
                            </label>
                            <div style={{ display: 'flex', gap: '8px' }}>
                                <input 
                                    ref={inputRef}
                                    type="text" 
                                    value={newLabel}
                                    onChange={(e) => setNewLabel(e.target.value)}
                                    style={{
                                        flex: 1,
                                        background: '#0d0d14',
                                        border: '1px solid rgba(255,255,255,0.1)',
                                        borderRadius: '6px',
                                        padding: '8px 12px',
                                        color: '#fff',
                                        fontSize: '14px',
                                        outline: 'none',
                                    }}
                                    placeholder="Enter new label..."
                                    disabled={addLabelMutation.isPending}
                                />
                                <button 
                                    type="submit"
                                    disabled={addLabelMutation.isPending || !newLabel.trim()}
                                    style={{
                                        padding: '8px 16px',
                                        background: '#10b981',
                                        border: 'none',
                                        borderRadius: '6px',
                                        color: '#fff',
                                        cursor: 'pointer',
                                        fontSize: '13px',
                                        fontWeight: 600,
                                        opacity: (addLabelMutation.isPending || !newLabel.trim()) ? 0.5 : 1,
                                    }}
                                >
                                    Add
                                </button>
                            </div>
                        </form>
                    </>
                )}

                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                    <button 
                        onClick={handleClose}
                        disabled={addLabelMutation.isPending || removeLabelMutation.isPending}
                        style={{
                            padding: '8px 16px',
                            background: '#3b82f6',
                            border: 'none',
                            borderRadius: '6px',
                            color: '#fff',
                            cursor: 'pointer',
                            fontSize: '13px',
                            fontWeight: 600,
                            opacity: (addLabelMutation.isPending || removeLabelMutation.isPending) ? 0.7 : 1,
                        }}
                    >
                        Done
                    </button>
                </div>
            </div>
        </div>
    );
}