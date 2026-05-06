interface CosmeticButtonProps {
    onClick: () => void;
    children: React.ReactNode;
    disabled?: boolean;
}

export default function CosmeticButton({ onClick, children, disabled = false }: CosmeticButtonProps) {
    return (
        <button
            onClick={onClick}
            disabled={disabled}
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
                cursor: disabled ? 'not-allowed' : 'pointer',
                opacity: disabled ? 0.5 : 1,
            }}
        >
            {children}
        </button>
    );
}