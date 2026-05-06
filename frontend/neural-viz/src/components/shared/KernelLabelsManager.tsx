import { useState } from 'react';
import KernelLabelsDialog from '../KernelLabelsDialog';
import CosmeticButton from './CosmeticButton';

interface KernelLabelsManagerProps {
    weightCoordinate: string;
}

export default function KernelLabelsManager({ weightCoordinate }: KernelLabelsManagerProps) {
    const [isLabelsDialogOpen, setIsLabelsDialogOpen] = useState(false);

    return (
        <>
            <CosmeticButton onClick={() => setIsLabelsDialogOpen(true)}>
                🏷️ Labels
            </CosmeticButton>

            <KernelLabelsDialog
                key={weightCoordinate}
                isOpen={isLabelsDialogOpen}
                weightCoordinate={weightCoordinate}
                onClose={() => setIsLabelsDialogOpen(false)}
            />
        </>
    );
}